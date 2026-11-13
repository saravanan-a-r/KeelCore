// The rewrite feature, independent of the keyboard it runs in: the installed model and its downloads, the engine
// host, and the controller that rewrites the field being edited. The keyboard app supplies only the UI around it.
import org.jetbrains.kotlin.gradle.dsl.JvmTarget

plugins {
    alias(libs.plugins.agp.library)
    alias(libs.plugins.kotlin.android)
}

val projectMinSdk: String by project
val projectCompileSdk: String by project

android {
    namespace = "com.echomebetter.rewrite"
    compileSdk = projectCompileSdk.toInt()

    defaultConfig {
        minSdk = projectMinSdk.toInt()
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    sourceSets {
        maybeCreate("main").java.srcDirs("src/main/kotlin")
        maybeCreate("test").java.srcDirs("src/test/kotlin")
    }

    buildTypes {
        // The app's build types, so each of its variants finds this module's.
        create("beta")
    }

    testOptions {
        unitTests.isIncludeAndroidResources = true
    }
}

kotlin {
    compilerOptions {
        jvmTarget.set(JvmTarget.JVM_17)
    }
}

dependencies {
    api(projects.modelstore)
    // The Android build of onnxruntime: the same Java API the engine compiles against, with the native libraries.
    implementation(libs.onnxruntime.android)
    implementation(libs.kotlinx.coroutines)
    implementation(libs.androidx.core.ktx)
    implementation(libs.androidx.work.runtime)

    testImplementation(libs.junit4)
    testImplementation(libs.robolectric)
    testImplementation(libs.androidx.test.core)
    testImplementation(libs.kotlinx.coroutines.test)
    testImplementation(kotlin("test-junit"))
}
