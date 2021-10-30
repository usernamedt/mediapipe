// Copyright 2019 The MediaPipe Authors.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//      http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.
//
// An example of sending OpenCV webcam frames into a MediaPipe graph.
#include <cmath>
#include <cstdlib>
#include <ctime>

#include "absl/flags/flag.h"
#include "absl/flags/parse.h"
#include "mediapipe/framework/calculator_framework.h"
#include "mediapipe/framework/formats/image_frame.h"
#include "mediapipe/framework/formats/image_frame_opencv.h"
#include "mediapipe/framework/port/file_helpers.h"
#include "mediapipe/framework/port/opencv_highgui_inc.h"
#include "mediapipe/framework/port/opencv_imgproc_inc.h"
#include "mediapipe/framework/port/opencv_video_inc.h"
#include "mediapipe/framework/port/parse_text_proto.h"
#include "mediapipe/framework/port/status.h"
#include "mediapipe/framework/formats/landmark.pb.h"

constexpr char kInputStream[] = "input_video";
constexpr char kOutputStream[] = "output_video";
constexpr char kPoseLandmarks[] = "pose_landmarks";
constexpr char kWindowName[] = "MediaPipe";

ABSL_FLAG(std::string, calculator_graph_config_file, "",
          "Name of file containing text format CalculatorGraphConfig proto.");
ABSL_FLAG(std::string, input_video_path, "",
          "Full path of video to load. "
          "If not provided, attempt to use a webcam.");
ABSL_FLAG(std::string, output_video_path, "",
          "Full path of where to save result (.mp4 only). "
          "If not provided, show result in a window.");

class GameState {
public:
    void detect_up() {
        auto currTime = time(nullptr);
        if (difftime(currTime, lastSquatTime) > 2 && wasDown) {
            wasDown = false;
            squatsCount += 1;
            lastSquatTime = currTime;
        }
    }

    void detect_down() {
        wasDown = true;
    }

    int get_total_squats() const {
        return squatsCount;
    }

private:
    time_t lastSquatTime;
    int squatsCount;
    bool wasDown;
};

void drawStats(cv::Mat frame, GameState *state, float lAngle, float rAngle) {
    int x = 0;
    int y = 0;
    int width = frame.cols;
    int height = 50;
    cv::Point pt1(x, y);
    cv::Point pt2(x + width, y + height);
    cv::Point ptt(x + 10, y + height / 2 + 10);
    cv::rectangle(frame, pt1, pt2, cv::Scalar(211, 211, 211), -1);
    cv::putText(frame,
                cv::format("You've done %d squats! Left angle %f, Right angle %f", state->get_total_squats(), lAngle,
                           rAngle),
                ptt,
                cv::FONT_HERSHEY_SIMPLEX,
                0.5,
                CV_RGB(0, 0, 0), //font color
                1);
}

void processSingleFrame(cv::Mat frame, GameState *state, float lAngle, float rAngle) {
    if (lAngle > 25 && rAngle > 25 && lAngle < 80 && rAngle < 80) {
        state->detect_down();
    }
    if (lAngle > 94 && rAngle > 94) {
        state->detect_up();
    }
    drawStats(frame, state, lAngle, rAngle);
}

float
getAngleABC(const mediapipe::NormalizedLandmark &a, mediapipe::NormalizedLandmark b, mediapipe::NormalizedLandmark c) {
    cv::Point3f v1 = {b.x() - a.x(), b.y() - a.y(), b.z() - a.z()};
    cv::Point3f v2 = {b.x() - c.x(), b.y() - c.y(), b.z() - c.z()};

    auto v1len = sqrt(v1.x * v1.x + v1.y * v1.y + v1.z * v1.z);
    cv::Point3f v1norm = {v1.x / v1len, v1.y / v1len, v1.z / v1len};

    auto v2len = sqrt(v2.x * v2.x + v2.y * v2.y + v2.z * v2.z);
    cv::Point3f v2norm = {v2.x / v2len, v2.y / v2len, v2.z / v2len};

    auto res = v1norm.x * v2norm.x + v1norm.y * v2norm.y + v1norm.z * v2norm.z;

    return acos(res) * (180.0 / M_PI);
}


absl::Status RunMPPGraph() {
    std::string calculator_graph_config_contents;
    MP_RETURN_IF_ERROR(mediapipe::file::GetContents(
            absl::GetFlag(FLAGS_calculator_graph_config_file),
            &calculator_graph_config_contents));
    LOG(INFO) << "Get calculator graph config contents: "
              << calculator_graph_config_contents;
    auto config = mediapipe::ParseTextProtoOrDie<mediapipe::CalculatorGraphConfig>(
            calculator_graph_config_contents);

    LOG(INFO) << "Initialize the calculator graph.";
    mediapipe::CalculatorGraph graph;
    MP_RETURN_IF_ERROR(graph.Initialize(config));

    LOG(INFO) << "Initialize the camera.";
    cv::VideoCapture capture;
    capture.open(0);

    RET_CHECK(capture.isOpened());

    cv::namedWindow(kWindowName, /*flags=WINDOW_AUTOSIZE*/ 1);
    capture.set(cv::CAP_PROP_FRAME_WIDTH, 640);
    capture.set(cv::CAP_PROP_FRAME_HEIGHT, 480);
    capture.set(cv::CAP_PROP_FPS, 30);

    LOG(INFO) << "Start running the calculator graph.";
    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller poller,
                     graph.AddOutputStreamPoller(kOutputStream));
    ASSIGN_OR_RETURN(mediapipe::OutputStreamPoller landmarkPoller,
                     graph.AddOutputStreamPoller(kPoseLandmarks));

    MP_RETURN_IF_ERROR(graph.StartRun({}));


    GameState state = GameState();
    LOG(INFO) << "Start grabbing and processing frames.";
    bool grab_frames = true;
    while (grab_frames) {
        // Capture opencv camera or video frame.
        cv::Mat camera_frame_raw;
        capture >> camera_frame_raw;
        if (camera_frame_raw.empty()) {
            LOG(INFO) << "Ignore empty frames from camera.";
            continue;
        }
        cv::Mat camera_frame;
        cv::cvtColor(camera_frame_raw, camera_frame, cv::COLOR_BGR2RGB);

        cv::flip(camera_frame, camera_frame, 1);

        auto input_frame = absl::make_unique<mediapipe::ImageFrame>(
                mediapipe::ImageFormat::SRGB, camera_frame.cols, camera_frame.rows,
                mediapipe::ImageFrame::kDefaultAlignmentBoundary);
        cv::Mat input_frame_mat = mediapipe::formats::MatView(input_frame.get());
        camera_frame.copyTo(input_frame_mat);

        size_t frame_timestamp_us =
                (double) cv::getTickCount() / (double) cv::getTickFrequency() * 1e6;
        MP_RETURN_IF_ERROR(graph.AddPacketToInputStream(
                kInputStream, mediapipe::Adopt(input_frame.release())
                        .At(mediapipe::Timestamp(frame_timestamp_us))));

        mediapipe::Packet packet;
        if (!poller.Next(&packet)) break;
        auto &output_frame = packet.Get<mediapipe::ImageFrame>();

        float lAngle = 0;
        float rAngle = 0;
        if (landmarkPoller.QueueSize() > 0) {
            // Get the graph result packet, or stop if that fails.
            mediapipe::Packet packetL;

            if (!landmarkPoller.Next(&packetL)) break;
            const auto &input = packetL.Get<mediapipe::NormalizedLandmarkList>();

            // left leg
            auto lU = input.landmark(24);
            auto lM = input.landmark(26);
            auto lB = input.landmark(28);
            lAngle = getAngleABC(lU, lM, lB);

            // right leg
            auto rU = input.landmark(23);
            auto rM = input.landmark(25);
            auto rB = input.landmark(27);
            rAngle = getAngleABC(lU, lM, lB);
        }

        cv::Mat output_frame_mat = mediapipe::formats::MatView(&output_frame);
        cv::cvtColor(output_frame_mat, output_frame_mat, cv::COLOR_RGB2BGR);

        processSingleFrame(output_frame_mat, &state, lAngle, rAngle);

        cv::imshow(kWindowName, output_frame_mat);

        const int pressed_key = cv::waitKey(5);
        if (pressed_key >= 0 && pressed_key != 255) grab_frames = false;
    }

    LOG(INFO) << "Shutdown";
    MP_RETURN_IF_ERROR(graph.CloseInputStream(kInputStream));
    return graph.WaitUntilDone();
}

int main(int argc, char **argv) {
    google::InitGoogleLogging(argv[0]);
    absl::ParseCommandLine(argc, argv);
    absl::Status run_status = RunMPPGraph();
    if (!run_status.ok()) {
        LOG(ERROR) << "Failed to run the graph: " << run_status.message();
        return EXIT_FAILURE;
    } else {
        LOG(INFO) << "Success!";
    }
    return EXIT_SUCCESS;
}
