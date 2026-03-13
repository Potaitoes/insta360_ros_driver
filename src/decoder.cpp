#include <iostream>
#include <thread>
#include <string>
#include <queue>
#include <mutex>
#include <condition_variable>
#include <atomic>

#include <opencv2/opencv.hpp>

#include "rclcpp/rclcpp.hpp"
#include "rclcpp/qos.hpp"
#include "sensor_msgs/msg/compressed_image.hpp"
#include "sensor_msgs/msg/image.hpp"
#include "cv_bridge/cv_bridge.hpp" 
#include "sensor_msgs/image_encodings.hpp"

extern "C" {
    #include <libavcodec/avcodec.h>
    #include <libavformat/avformat.h>
    #include <libswscale/swscale.h>
    #include <libavutil/imgutils.h>
}

class H264DecoderNode : public rclcpp::Node {
private:
    const AVCodec* codec_ = nullptr;
    AVCodecContext* codec_ctx_ = nullptr;
    AVCodecParserContext* parser_ctx_ = nullptr;
    AVPacket* pkt_ = nullptr;
    AVFrame* hw_frame_ = nullptr;
    SwsContext* sws_ctx_ = nullptr;
    cv::Mat bgr_frame_; 

    rclcpp::Subscription<sensor_msgs::msg::CompressedImage>::SharedPtr subscription_;
    rclcpp::Publisher<sensor_msgs::msg::Image>::SharedPtr publisher_;

    std::thread publisher_thread_;
    std::queue<cv::Mat> frame_publish_queue_;
    std::mutex queue_mutex_;
    std::condition_variable queue_cv_;
    std::atomic<bool> stop_publisher_thread_{false};
    size_t max_queue_size_ = 10;
    
    int skip_frame_ = 0;
    int frame_counter_ = 0;
    bool i_frame_only_ = false;

    void InitFFmpegDecoder() {
        // Force Software H.264 Decoder for NUC compatibility
        codec_ = avcodec_find_decoder(AV_CODEC_ID_H264);
        if (!codec_) {
            RCLCPP_ERROR(this->get_logger(), "No H.264 decoder available!");
            return;
        }

        parser_ctx_ = av_parser_init(codec_->id);
        codec_ctx_ = avcodec_alloc_context3(codec_);
        
        if (!codec_ctx_) {
            RCLCPP_ERROR(this->get_logger(), "Failed to allocate codec context");
            return;
        }

        if (avcodec_open2(codec_ctx_, codec_, nullptr) < 0) {
            RCLCPP_ERROR(this->get_logger(), "Failed to open codec");
            return;
        }

        pkt_ = av_packet_alloc();
        hw_frame_ = av_frame_alloc();
    }

    void PublisherThreadLoop() {
        while (!stop_publisher_thread_) {
            cv::Mat frame_to_publish;
            {
                std::unique_lock<std::mutex> lock(queue_mutex_);
                queue_cv_.wait(lock, [this] {
                    return !frame_publish_queue_.empty() || stop_publisher_thread_;
                });

                if (stop_publisher_thread_ && frame_publish_queue_.empty()) break;
                
                frame_to_publish = frame_publish_queue_.front();
                frame_publish_queue_.pop();
            }

            if (!frame_to_publish.empty()) {
                auto img_msg = std::make_unique<sensor_msgs::msg::Image>();
                std_msgs::msg::Header header;
                header.stamp = this->get_clock()->now();
                header.frame_id = "camera_frame";
                
                cv_bridge::CvImage cv_image(header, sensor_msgs::image_encodings::BGR8, frame_to_publish);
                cv_image.toImageMsg(*img_msg);
                publisher_->publish(std::move(img_msg));
            }
        }
    }

    void DecodeAndDisplayPacket(AVPacket* packet) {
        if (avcodec_send_packet(codec_ctx_, packet) < 0) return;

        while (true) {
            int ret = avcodec_receive_frame(codec_ctx_, hw_frame_);
            if (ret == AVERROR(EAGAIN) || ret == AVERROR_EOF) break;
            if (ret < 0) return;

            // In software mode, the frame is directly in system memory
            AVFrame* frame_to_display = hw_frame_;

            if (!sws_ctx_ && frame_to_display->width > 0) {
                sws_ctx_ = sws_getContext(
                    frame_to_display->width, frame_to_display->height, (AVPixelFormat)frame_to_display->format,
                    frame_to_display->width, frame_to_display->height, AV_PIX_FMT_BGR24,
                    SWS_FAST_BILINEAR, nullptr, nullptr, nullptr);
                bgr_frame_.create(frame_to_display->height, frame_to_display->width, CV_8UC3);
            }

            if (sws_ctx_) {
                uint8_t* dst_data[4] = { bgr_frame_.data, nullptr, nullptr, nullptr };
                int dst_linesize[4] = { static_cast<int>(bgr_frame_.step[0]), 0, 0, 0 };
                sws_scale(sws_ctx_, frame_to_display->data, frame_to_display->linesize,
                          0, frame_to_display->height, dst_data, dst_linesize);

                if (skip_frame_ == 0 || (frame_counter_++ % (skip_frame_ + 1) == 0)) {
                    cv::Mat frame_copy = bgr_frame_.clone();
                    std::lock_guard<std::mutex> lock(queue_mutex_);
                    if (frame_publish_queue_.size() < max_queue_size_) {
                        frame_publish_queue_.push(frame_copy);
                        queue_cv_.notify_one();
                    }
                }
            }
            av_frame_unref(hw_frame_);
        }
    }

    void CleanupFFmpegDecoder() {
        if (sws_ctx_) sws_freeContext(sws_ctx_);
        if (hw_frame_) av_frame_free(&hw_frame_);
        if (pkt_) av_packet_free(&pkt_);
        if (codec_ctx_) avcodec_free_context(&codec_ctx_);
        if (parser_ctx_) av_parser_close(parser_ctx_);
    }

    void compressed_image_callback(const sensor_msgs::msg::CompressedImage::SharedPtr msg) {
        if (msg->format != "h264" || !codec_ctx_) return;

        const uint8_t* cur_data = msg->data.data();
        size_t remaining_size = msg->data.size();

        while (remaining_size > 0) {
            int bytes_parsed = av_parser_parse2(parser_ctx_, codec_ctx_,
                                                &pkt_->data, &pkt_->size,
                                                cur_data, static_cast<int>(remaining_size),
                                                AV_NOPTS_VALUE, AV_NOPTS_VALUE, 0);
            cur_data += bytes_parsed;
            remaining_size -= bytes_parsed;

            if (pkt_->size > 0) {
                if (!i_frame_only_ || parser_ctx_->key_frame == 1) {
                    DecodeAndDisplayPacket(pkt_);
                }
            }
        }
    }

public:
    H264DecoderNode() : Node("h264_decoder_node") {
        this->declare_parameter("compressed_topic", "/dual_fisheye/image/compressed");
        this->declare_parameter("uncompressed_topic", "/dual_fisheye/image");
        this->declare_parameter("skip_frame", 0);
        this->declare_parameter("i_frame_only", false);

        auto sub_topic = this->get_parameter("compressed_topic").as_string();
        auto pub_topic = this->get_parameter("uncompressed_topic").as_string();
        skip_frame_ = this->get_parameter("skip_frame").as_int();
        i_frame_only_ = this->get_parameter("i_frame_only").as_bool();

        subscription_ = this->create_subscription<sensor_msgs::msg::CompressedImage>(
            sub_topic, rclcpp::SensorDataQoS(),
            std::bind(&H264DecoderNode::compressed_image_callback, this, std::placeholders::_1));

        publisher_ = this->create_publisher<sensor_msgs::msg::Image>(pub_topic, 10);
        publisher_thread_ = std::thread(&H264DecoderNode::PublisherThreadLoop, this);
        
        InitFFmpegDecoder();
    }

    ~H264DecoderNode() {
        stop_publisher_thread_ = true;
        queue_cv_.notify_one();
        if (publisher_thread_.joinable()) publisher_thread_.join();
        CleanupFFmpegDecoder();
    }
};

int main(int argc, char* argv[]) {
    rclcpp::init(argc, argv);
    rclcpp::spin(std::make_shared<H264DecoderNode>());
    rclcpp::shutdown();
    return 0;
}
