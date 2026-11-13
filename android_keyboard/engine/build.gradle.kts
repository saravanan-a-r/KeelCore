// The Thelivu rewrite engine, ported from the Chrome extension (chrome_extension/src/engine) and held to the same
// fixtures. Plain Kotlin/JVM: it compiles against the onnxruntime Java API, which the app supplies at runtime from
// the Android build of onnxruntime, and its tests run on the desktop build.
plugins {
    alias(libs.plugins.kotlin.jvm)
}

kotlin {
    jvmToolchain(17)
}

dependencies {
    api(libs.kotlinx.coroutines.core)
    implementation(libs.kotlinx.serialization.json)
    compileOnly(libs.onnxruntime.jvm)

    testImplementation(libs.onnxruntime.jvm)
    testImplementation(libs.junit4)
    testImplementation(libs.kotlinx.coroutines.test)
    testImplementation(kotlin("test-junit"))
}

tasks.test {
    // Opt-in check against the released model: THELIVU_MODEL_DIR=<folder with model.json, base and adapters>.
    System.getenv("THELIVU_MODEL_DIR")?.let { environment("THELIVU_MODEL_DIR", it) }
    maxHeapSize = "2g"
}
