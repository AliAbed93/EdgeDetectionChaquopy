# Lens Scanner ProGuard Rules

# Keep Chaquopy classes
-keep class com.chaquo.python.** { *; }
-keep class org.python.** { *; }

# Keep Python module entry points
-keepclassmembers class * {
    @com.chaquo.python.PyMethod *;
}
