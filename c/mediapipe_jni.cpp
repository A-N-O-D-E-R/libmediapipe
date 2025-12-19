#include <jni.h>
#include <string>
#include <mediapipe.h>

struct FaceContext {
    mp_instance* instance;
    mp_poller* landmarks_poller;
};

extern "C" {

// --------------------------------------------------
// Create Face graph
// --------------------------------------------------

JNIEXPORT jlong JNICALL
Java_fr_audioptic_mediapipe_jni_MediaPipeFaceNative_create(
        JNIEnv* env,
        jclass,
        jstring resourceDir) {

    const char* dir = env->GetStringUTFChars(resourceDir, nullptr);
    mp_set_resource_dir(dir);

    std::string graph =
        std::string(dir) +
        "/mediapipe/modules/face_landmark/"
        "face_landmark_front_cpu.binarypb";

    mp_instance_builder* builder =
        mp_create_instance_builder(graph.c_str(), "image");

    // Optional tuning
    mp_add_option_float(
        builder,
        "FaceDetectionShortRangeCpu__TensorsToDetectionsCalculator",
        "min_score_thresh",
        0.5f);

    mp_add_side_packet(builder, "num_faces", mp_create_packet_int(1));
    mp_add_side_packet(builder, "with_attention", mp_create_packet_bool(false));

    mp_instance* instance = mp_create_instance(builder);
    if (!instance) return 0;

    mp_poller* poller =
        mp_create_poller(instance, "multi_face_landmarks");
    if (!poller) return 0;

    if (!mp_start(instance)) return 0;

    auto* ctx = new FaceContext{instance, poller};
    env->ReleaseStringUTFChars(resourceDir, dir);

    return reinterpret_cast<jlong>(ctx);
}

// --------------------------------------------------
// Process image and return FACE landmarks
// --------------------------------------------------

JNIEXPORT jobjectArray JNICALL
Java_fr_audioptic_mediapipe_jni_MediaPipeFaceNative_processImage(
        JNIEnv* env,
        jclass,
        jlong handle,
        jbyteArray rgb,
        jint width,
        jint height) {

    auto* ctx = reinterpret_cast<FaceContext*>(handle);
    if (!ctx) return nullptr;

    jbyte* data = env->GetByteArrayElements(rgb, nullptr);

    mp_image image;
    image.data = reinterpret_cast<uint8_t*>(data);
    image.width = width;
    image.height = height;
    image.format = mp_image_format_srgb;

    mp_process(ctx->instance, mp_create_packet_image(image));
    mp_wait_until_idle(ctx->instance);

    env->ReleaseByteArrayElements(rgb, data, JNI_ABORT);

    if (mp_get_queue_size(ctx->landmarks_poller) == 0) {
        return nullptr;
    }

    mp_packet* packet = mp_poll_packet(ctx->landmarks_poller);
    mp_multi_face_landmark_list* faces =
        mp_get_norm_multi_face_landmarks(packet);

    // Java classes
    jclass landmarkCls =
        env->FindClass("fr/audioptic/mediapipe/jni/MediaPipeFaceNative$Landmark");
    jmethodID landmarkCtor =
        env->GetMethodID(landmarkCls, "<init>", "(FFF)V");

    jclass faceCls =
        env->FindClass("fr/audioptic/mediapipe/jni/MediaPipeFaceNative$FaceLandmarks");
    jmethodID faceCtor =
        env->GetMethodID(faceCls, "<init>",
            "([Lfr/audioptic/mediapipe/jni/MediaPipeFaceNative$Landmark;)V");

    jobjectArray result =
        env->NewObjectArray(faces->length, faceCls, nullptr);

    for (int i = 0; i < faces->length; ++i) {
        mp_landmark_list& list = faces->elements[i];

        jobjectArray landmarks =
            env->NewObjectArray(list.length, landmarkCls, nullptr);

        for (int j = 0; j < list.length; ++j) {
            mp_landmark& lm = list.elements[j];
            jobject jlm = env->NewObject(
                landmarkCls, landmarkCtor,
                lm.x, lm.y, lm.z);
            env->SetObjectArrayElement(landmarks, j, jlm);
        }

        jobject faceObj =
            env->NewObject(faceCls, faceCtor, landmarks);
        env->SetObjectArrayElement(result, i, faceObj);
    }

    mp_destroy_multi_face_landmarks(faces);
    mp_destroy_packet(packet);

    return result;
}

// --------------------------------------------------
// Cleanup
// --------------------------------------------------

JNIEXPORT void JNICALL
Java_fr_audioptic_mediapipe_jni_MediaPipeFaceNative_destroy(
        JNIEnv*,
        jclass,
        jlong handle) {

    auto* ctx = reinterpret_cast<FaceContext*>(handle);
    if (!ctx) return;

    mp_destroy_poller(ctx->landmarks_poller);
    mp_destroy_instance(ctx->instance);
    delete ctx;
}

}
