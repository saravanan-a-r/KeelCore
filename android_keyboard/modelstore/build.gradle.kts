// The model catalogue (model.json and each adapter's adapter.json), verified resumable downloads, and the store of
// installed files. Plain Kotlin/JVM: the app supplies the folder and schedules the work.
plugins {
    alias(libs.plugins.kotlin.jvm)
}

kotlin {
    jvmToolchain(17)
}

dependencies {
    api(project(":engine"))
    implementation(libs.kotlinx.serialization.json)

    testImplementation(libs.junit4)
    testImplementation(libs.kotlinx.coroutines.test)
    testImplementation(kotlin("test-junit"))
}

// The engine's fixture model (the extension's tiny test model) is what the installer tests serve over HTTP.
sourceSets.test {
    resources.srcDir("../engine/src/test/resources")
}
