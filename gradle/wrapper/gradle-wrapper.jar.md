# Gradle Wrapper JAR

The `gradle-wrapper.jar` file should be downloaded automatically when you run:

```bash
# On Windows
gradlew.bat tasks

# On Linux/Mac
./gradlew tasks
```

If you need to manually generate the wrapper, run from Android Studio terminal:
```bash
gradle wrapper --gradle-version 8.4
```

This will create the `gradle-wrapper.jar` file in this directory.
