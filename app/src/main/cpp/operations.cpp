#include <jni.h>
#include <string>
#include <cmath>

float L2Norm( float* x , float* y , int& size ) {
    float* baseAddress1 = x ;
    float* baseAddress2 = y ;
    int i = 0 ;
    float squaredSum = 0.0f ;
    while( i < size ) {
        squaredSum += ( *baseAddress1 - *baseAddress2 ) * ( *baseAddress1 - *baseAddress2 ) ;
        baseAddress1++ ;
        baseAddress2++ ;
        i++ ;
    }
    return sqrt( squaredSum ) ;
}

float cosineSimilarity( float* x , float* y , int& size ){
    float* elementAddressX = x ;
    float* elementAddressY = y ;
    int i = 0 ;
    float squaredSumX = 0, squaredSumY = 0, dotProduct = 0;
    while( i < size ) {
        dotProduct += *elementAddressX * *elementAddressY ;
        squaredSumX += ( *elementAddressX * *elementAddressX ) ;
        squaredSumY += ( *elementAddressY * *elementAddressY ) ;
        elementAddressX++ ;
        elementAddressY++ ;
        i++ ;
    }
    return dotProduct / (sqrtf(squaredSumX) * sqrtf(squaredSumY)) ;
}

extern "C"
JNIEXPORT jfloat JNICALL Java_com_ml_quaterion_facenetdetection_FrameAnalyser_nativeL2Norm(
        JNIEnv *env,
        jobject type,
        jfloatArray x,
        jfloatArray y,
        jint size) {
    jfloat* baseAddressX = env -> GetFloatArrayElements( x , nullptr );
    jfloat* baseAddressY = env -> GetFloatArrayElements( y , nullptr );
    int i = 0 ;
    float squaredSum = 0.0f ;
    while( i < size ) {
        squaredSum += ( *baseAddressX - *baseAddressY ) * ( *baseAddressX - *baseAddressY ) ;
        baseAddressX++ ;
        baseAddressY++ ;
        i++ ;
    }
    env -> ReleaseFloatArrayElements( x , baseAddressX , 0);
    env -> ReleaseFloatArrayElements( y , baseAddressY , 0);
    return sqrt( squaredSum ) ; ;
}

extern "C"
JNIEXPORT jfloat JNICALL Java_com_ml_quaterion_facenetdetection_FrameAnalyser_nativeCosineSimilarity(
        JNIEnv *env,
        jobject type,
        jfloatArray x,
        jfloatArray y,
        jint size) {
    jfloat* baseAddressX = env -> GetFloatArrayElements( x , nullptr );
    jfloat* baseAddressY = env -> GetFloatArrayElements( y , nullptr );
    float score = cosineSimilarity( baseAddressX , baseAddressY , size ) ;
    env -> ReleaseFloatArrayElements( x , baseAddressX , 0);
    env -> ReleaseFloatArrayElements( y , baseAddressY , 0);
    return score ;
}


extern "C"
JNIEXPORT jfloat JNICALL Java_com_ml_quaterion_facenetdetection_FrameAnalyser_averageL2Cluster(
        JNIEnv *env,
        jobject type,
        jfloatArray subjectEmbedding,
        jint embeddingSize ,
        jobjectArray cluster,
        jint clusterSize) {

    jfloat* subjectAddr = env -> GetFloatArrayElements( subjectEmbedding , nullptr );

    float scoreSum = 0.0f ;
    for( int i = 0 ; i < clusterSize ; i++ ) {
        auto clusterEmbedding = (jfloatArray) env -> GetObjectArrayElement( cluster , i );
        jfloat* clusterAddr = env -> GetFloatArrayElements( clusterEmbedding , nullptr );
        scoreSum += L2Norm( subjectAddr , clusterAddr , embeddingSize ) ;
        env -> DeleteLocalRef( clusterEmbedding ) ;
    }

    env -> ReleaseFloatArrayElements( subjectEmbedding , subjectAddr , 0);
    // env -> ReleaseFloatArrayElements( cluster , clusterAddr , 0);

    return scoreSum / (float)clusterSize ;
}










