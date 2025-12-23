package com.lensscanner.app

import android.Manifest
import android.content.pm.PackageManager
import android.graphics.Bitmap
import android.graphics.BitmapFactory
import android.net.Uri
import android.os.Bundle
import android.util.Log
import android.view.View
import android.widget.Toast
import androidx.activity.result.contract.ActivityResultContracts
import androidx.appcompat.app.AppCompatActivity
import androidx.camera.core.*
import androidx.camera.lifecycle.ProcessCameraProvider
import androidx.core.content.ContextCompat
import androidx.lifecycle.lifecycleScope
import com.chaquo.python.PyObject
import com.chaquo.python.Python
import com.chaquo.python.android.AndroidPlatform
import com.lensscanner.app.databinding.ActivityMainBinding
import kotlinx.coroutines.Dispatchers
import kotlinx.coroutines.launch
import kotlinx.coroutines.withContext
import java.io.ByteArrayOutputStream
import java.io.File
import java.text.SimpleDateFormat
import java.util.*
import java.util.concurrent.ExecutorService
import java.util.concurrent.Executors

/**
 * Main Activity for Lens Scanner application.
 * 
 * Provides UI for:
 * - Camera preview and capture
 * - Image loading from gallery
 * - Lens scanning and SVG export
 */
class MainActivity : AppCompatActivity() {

    private lateinit var binding: ActivityMainBinding
    private lateinit var cameraExecutor: ExecutorService
    
    private var imageCapture: ImageCapture? = null
    private var currentImageUri: Uri? = null
    private var currentImageBytes: ByteArray? = null
    
    // Python module reference
    private var lensScannerModule: PyObject? = null
    
    companion object {
        private const val TAG = "LensScanner"
        private const val MARKER_DISTANCE_MM = 100.0 // Default marker distance
    }
    
    // Permission request launcher
    private val requestPermissionLauncher = registerForActivityResult(
        ActivityResultContracts.RequestPermission()
    ) { isGranted ->
        if (isGranted) {
            startCamera()
        } else {
            Toast.makeText(this, "Camera permission required", Toast.LENGTH_SHORT).show()
        }
    }
    
    // Image picker launcher
    private val pickImageLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri ->
        uri?.let { loadImageFromUri(it) }
    }

    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)
        binding = ActivityMainBinding.inflate(layoutInflater)
        setContentView(binding.root)
        
        // Initialize Python
        initializePython()
        
        // Initialize camera executor
        cameraExecutor = Executors.newSingleThreadExecutor()
        
        // Setup UI
        setupUI()
        
        // Check camera permission
        checkCameraPermission()
    }
    
    private fun initializePython() {
        // Start Python if not already started
        if (!Python.isStarted()) {
            Python.start(AndroidPlatform(this))
        }
        
        // Get reference to lens scanner module
        val py = Python.getInstance()
        lensScannerModule = py.getModule("lens_scanner.pipeline")
        
        // Log version
        val version = lensScannerModule?.callAttr("get_version")?.toString()
        Log.i(TAG, "Lens Scanner Pipeline version: $version")
    }
    
    private fun setupUI() {
        // Scan button - process current image
        binding.btnScan.setOnClickListener {
            processCurrentImage()
        }
        
        // Capture button - take photo from camera
        binding.btnCapture.setOnClickListener {
            captureImage()
        }
        
        // Load button - pick image from gallery
        binding.btnLoad.setOnClickListener {
            pickImageLauncher.launch("image/*")
        }
    }
    
    private fun checkCameraPermission() {
        when {
            ContextCompat.checkSelfPermission(
                this, Manifest.permission.CAMERA
            ) == PackageManager.PERMISSION_GRANTED -> {
                startCamera()
            }
            else -> {
                requestPermissionLauncher.launch(Manifest.permission.CAMERA)
            }
        }
    }
    
    private fun startCamera() {
        val cameraProviderFuture = ProcessCameraProvider.getInstance(this)
        
        cameraProviderFuture.addListener({
            val cameraProvider = cameraProviderFuture.get()
            
            // Preview use case
            val preview = Preview.Builder()
                .build()
                .also {
                    it.setSurfaceProvider(binding.cameraPreview.surfaceProvider)
                }
            
            // Image capture use case
            imageCapture = ImageCapture.Builder()
                .setCaptureMode(ImageCapture.CAPTURE_MODE_MAXIMIZE_QUALITY)
                .build()
            
            // Select back camera
            val cameraSelector = CameraSelector.DEFAULT_BACK_CAMERA
            
            try {
                // Unbind all use cases before rebinding
                cameraProvider.unbindAll()
                
                // Bind use cases to camera
                cameraProvider.bindToLifecycle(
                    this,
                    cameraSelector,
                    preview,
                    imageCapture
                )
                
                Log.i(TAG, "Camera started successfully")
                
            } catch (e: Exception) {
                Log.e(TAG, "Camera binding failed", e)
            }
            
        }, ContextCompat.getMainExecutor(this))
    }
    
    private fun captureImage() {
        val imageCapture = imageCapture ?: return
        
        showLoading(true, "Capturing image...")
        
        // Create output file
        val photoFile = createImageFile()
        val outputOptions = ImageCapture.OutputFileOptions.Builder(photoFile).build()
        
        imageCapture.takePicture(
            outputOptions,
            ContextCompat.getMainExecutor(this),
            object : ImageCapture.OnImageSavedCallback {
                override fun onImageSaved(output: ImageCapture.OutputFileResults) {
                    currentImageUri = Uri.fromFile(photoFile)
                    loadImageFromUri(currentImageUri!!)
                    showLoading(false)
                    Log.i(TAG, "Image captured: ${photoFile.absolutePath}")
                }
                
                override fun onError(exception: ImageCaptureException) {
                    showLoading(false)
                    Log.e(TAG, "Image capture failed", exception)
                    showError("Failed to capture image: ${exception.message}")
                }
            }
        )
    }
    
    private fun loadImageFromUri(uri: Uri) {
        lifecycleScope.launch {
            showLoading(true, "Loading image...")
            
            try {
                val bytes = withContext(Dispatchers.IO) {
                    contentResolver.openInputStream(uri)?.use { it.readBytes() }
                }
                
                if (bytes != null) {
                    currentImageBytes = bytes
                    currentImageUri = uri
                    
                    // Display preview
                    val bitmap = BitmapFactory.decodeByteArray(bytes, 0, bytes.size)
                    binding.imagePreview.setImageBitmap(bitmap)
                    binding.imagePreview.visibility = View.VISIBLE
                    binding.cameraPreview.visibility = View.GONE
                    
                    // Enable scan button
                    binding.btnScan.isEnabled = true
                    
                    Log.i(TAG, "Image loaded: ${bytes.size} bytes")
                } else {
                    showError("Failed to read image")
                }
                
            } catch (e: Exception) {
                Log.e(TAG, "Failed to load image", e)
                showError("Failed to load image: ${e.message}")
            }
            
            showLoading(false)
        }
    }
    
    private fun processCurrentImage() {
        val imageBytes = currentImageBytes
        if (imageBytes == null) {
            showError("No image loaded")
            return
        }
        
        lifecycleScope.launch {
            showLoading(true, "Processing lens contour...")
            
            try {
                val result = withContext(Dispatchers.Default) {
                    scanLensFromBytes(imageBytes)
                }
                
                handleScanResult(result)
                
            } catch (e: Exception) {
                Log.e(TAG, "Scan failed", e)
                showError("Scan failed: ${e.message}")
            }
            
            showLoading(false)
        }
    }
    
    /**
     * Call Python lens scanner pipeline.
     */
    private fun scanLensFromBytes(imageBytes: ByteArray): Map<String, Any?> {
        val module = lensScannerModule 
            ?: throw IllegalStateException("Python module not initialized")
        
        // Create output directory
        val outputDir = File(filesDir, "lens_scans").apply { mkdirs() }
        
        // Generate unique filename
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val outputFilename = "lens_$timestamp.svg"
        
        // Call Python function
        val pyResult = module.callAttr(
            "scan_lens_from_bytes",
            imageBytes,
            outputDir.absolutePath,
            MARKER_DISTANCE_MM
        )
        
        // Convert PyObject to Kotlin Map
        return pyObjectToMap(pyResult)
    }
    
    /**
     * Convert Python dictionary to Kotlin Map.
     */
    private fun pyObjectToMap(pyObj: PyObject): Map<String, Any?> {
        val map = mutableMapOf<String, Any?>()
        
        val pyDict = pyObj.asMap()
        for ((key, value) in pyDict) {
            val keyStr = key.toString()
            map[keyStr] = when {
                value == null -> null
                value.toString() == "True" -> true
                value.toString() == "False" -> false
                else -> {
                    try {
                        // Try to convert nested dict
                        pyObjectToMap(value)
                    } catch (e: Exception) {
                        // Fall back to string/number conversion
                        value.toString().toDoubleOrNull() ?: value.toString()
                    }
                }
            }
        }
        
        return map
    }
    
    private fun handleScanResult(result: Map<String, Any?>) {
        val success = result["success"] as? Boolean ?: false
        
        if (success) {
            val svgPath = result["svg_path"] as? String ?: ""
            val processingTime = result["processing_time_ms"] as? Double ?: 0.0
            
            // Get contour info
            @Suppress("UNCHECKED_CAST")
            val contourInfo = result["contour"] as? Map<String, Any?>
            val perimeter = contourInfo?.get("perimeter_mm") as? Double ?: 0.0
            val area = contourInfo?.get("area_mm2") as? Double ?: 0.0
            
            // Show success message
            val message = buildString {
                append("Scan complete!\n")
                append("Perimeter: %.2f mm\n".format(perimeter))
                append("Area: %.2f mm²\n".format(area))
                append("Time: %.0f ms\n".format(processingTime))
                append("SVG: $svgPath")
            }
            
            binding.tvResult.text = message
            binding.tvResult.visibility = View.VISIBLE
            
            Log.i(TAG, "Scan successful: $svgPath")
            Toast.makeText(this, "SVG saved!", Toast.LENGTH_SHORT).show()
            
        } else {
            val errorMessage = result["error_message"] as? String ?: "Unknown error"
            showError(errorMessage)
        }
    }
    
    private fun createImageFile(): File {
        val timestamp = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.US).format(Date())
        val storageDir = File(filesDir, "captures").apply { mkdirs() }
        return File(storageDir, "IMG_$timestamp.jpg")
    }
    
    private fun showLoading(show: Boolean, message: String = "") {
        binding.progressBar.visibility = if (show) View.VISIBLE else View.GONE
        binding.tvLoading.text = message
        binding.tvLoading.visibility = if (show && message.isNotEmpty()) View.VISIBLE else View.GONE
        
        // Disable buttons while loading
        binding.btnScan.isEnabled = !show && currentImageBytes != null
        binding.btnCapture.isEnabled = !show
        binding.btnLoad.isEnabled = !show
    }
    
    private fun showError(message: String) {
        binding.tvResult.text = "Error: $message"
        binding.tvResult.visibility = View.VISIBLE
        Toast.makeText(this, message, Toast.LENGTH_LONG).show()
        Log.e(TAG, message)
    }
    
    override fun onDestroy() {
        super.onDestroy()
        cameraExecutor.shutdown()
    }
}
