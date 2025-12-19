package(default_visibility = ["//visibility:public"])

cc_library(
    name = "mediapipe_c_api",
    srcs = ["mediapipe.cpp"],
    hdrs = ["mediapipe.h"],
    defines = ["MEDIAPIPE_EXPORT"],
    visibility = ["//visibility:public"],
    alwayslink = 1,
    deps = [
        "//mediapipe/framework:calculator_framework",
        "//mediapipe/framework/formats:image_frame",
        "//mediapipe/framework/formats:image_frame_opencv",
        "//mediapipe/framework/port:file_helpers",
        "//mediapipe/framework/port:opencv_highgui",
        "//mediapipe/framework/port:opencv_imgproc",
        "//mediapipe/framework/port:opencv_video",
        "//mediapipe/framework/port:parse_text_proto",
        "//mediapipe/framework/port:status",
        "@com_google_absl//absl/flags:flag",
        "@com_google_absl//absl/flags:parse",
        "//mediapipe/graphs/face_mesh:desktop_live_calculators",
        "//mediapipe/modules/hand_landmark:hand_landmark_tracking_cpu",
    ],
)

cc_binary(
    name = "mediapipe",
    deps = [":mediapipe_c_api"],
    data = [
        "//mediapipe/modules/face_detection:face_detection_short_range.tflite",
        "//mediapipe/modules/face_landmark:face_landmark_with_attention.tflite",
        "//mediapipe/modules/hand_landmark:hand_landmark_full.tflite",
        "//mediapipe/modules/palm_detection:palm_detection_full.tflite",
    ],
    linkshared = 1,
    linkopts = ["-Wl,--export-dynamic"],
)

genrule(
    name = "copy_jni_h",
    srcs = ["@bazel_tools//tools/jdk:jni_header"],
    outs = ["jni.h"],
    cmd = "cp -f $< $@",
)

genrule(
    name = "copy_jni_md_h",
    srcs = select({
        "//mediapipe:macos": ["@bazel_tools//tools/jdk:jni_md_header-darwin"],
        "//conditions:default": ["@bazel_tools//tools/jdk:jni_md_header-linux"],
    }),
    outs = ["jni_md.h"],
    cmd = "cp -f $< $@",
)

cc_library(
    name = "jni_headers",
    hdrs = [
        ":jni.h",
        ":jni_md.h",
    ],
    includes = ["."],
    visibility = ["//visibility:public"],
)

cc_binary(
    name = "libmediapipe_jni.so",
    srcs = ["mediapipe_jni.cpp"],
    deps = [
        ":mediapipe_c_api",
        ":jni_headers",
    ],
    linkshared = 1,
    linkopts = ["-Wl,--export-dynamic"],
)
