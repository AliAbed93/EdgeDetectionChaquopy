plugins {
    id("com.android.application")
    id("org.jetbrains.kotlin.android")
    id("com.chaquo.python")
}

android {
    namespace = "com.lensscanner.app"
    compileSdk = 34

    defaultConfig {
        applicationId = "com.lensscanner.app"
        minSdk = 24
        targetSdk = 34
        versionCode = 1
        versionName = "1.0.0-MVP"

        // Chaquopy configuration
        ndk {
            abiFilters += listOf("arm64-v8a", "armeabi-v7a", "x86_64")
        }

        python {
            // Python version
            version = "3.11"

            // Python packages for computer vision pipeline
            pip {
                install("numpy")
                install("opencv-python")
                install("scipy")
                install("svgwrite")
            }
        }
    }

    buildTypes {
        release {
            isMinifyEnabled = false
            proguardFiles(
                getDefaultProguardFile("proguard-android-optimize.txt"),
                "proguard-rules.pro"
            )
        }
    }

    compileOptions {
        sourceCompatibility = JavaVersion.VERSION_17
        targetCompatibility = JavaVersion.VERSION_17
    }

    kotlinOptions {
        jvmTarget = "17"
    }

    buildFeatures {
        viewBinding = true
    }

    // Source sets for Python code
    sourceSets {
        getByName("main") {
            python.srcDir("src/main/python")
        }
    }
}

dependencies {
    implementation("androidx.core:core-ktx:1.12.0")
    implementation("androidx.appcompat:appcompat:1.6.1")
    implementation("com.google.android.material:material:1.11.0")
    implementation("androidx.constraintlayout:constraintlayout:2.1.4")
    implementation("androidx.activity:activity-ktx:1.8.2")
    implementation("androidx.lifecycle:lifecycle-runtime-ktx:2.7.0")
    
    // CameraX for camera capture
    val cameraxVersion = "1.3.1"
    implementation("androidx.camera:camera-core:$cameraxVersion")
    implementation("androidx.camera:camera-camera2:$cameraxVersion")
    implementation("androidx.camera:camera-lifecycle:$cameraxVersion")
    implementation("androidx.camera:camera-view:$cameraxVersion")
    
    // Coroutines for async processing
    implementation("org.jetbrains.kotlinx:kotlinx-coroutines-android:1.7.3")
}
