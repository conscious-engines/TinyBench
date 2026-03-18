package ai.mlc.mlcchat

import android.Manifest
import android.content.Intent
import android.content.ContentValues
import android.content.pm.PackageManager
import android.net.Uri
import android.os.Build
import android.os.Bundle
import android.os.Handler
import android.os.Looper
import android.provider.MediaStore
import android.util.Log
import android.app.KeyguardManager
import android.content.Context
import android.view.WindowManager
import androidx.activity.ComponentActivity
import androidx.activity.compose.setContent
import androidx.activity.result.contract.ActivityResultContracts
import androidx.annotation.RequiresApi
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.material3.ExperimentalMaterial3Api
import androidx.compose.material3.Surface
import androidx.compose.ui.Modifier
import androidx.core.content.ContextCompat
import ai.mlc.mlcchat.ui.theme.MLCChatTheme
import java.text.SimpleDateFormat
import java.util.Date
import java.util.Locale
import java.util.UUID

class MainActivity : ComponentActivity() {
    var hasImage = false
    lateinit var chatState: AppViewModel.ChatState

    private val pickImageLauncher = registerForActivityResult(
        ActivityResultContracts.GetContent()
    ) { uri: Uri? ->
        uri?.let {
            Log.v("pickImageLauncher", "Selected image uri: $it")
        }
    }

    private var cameraImageUri: Uri? = null
    private val takePictureLauncher = registerForActivityResult(
        ActivityResultContracts.TakePicture()
    ) { success: Boolean ->
        if (success && cameraImageUri != null) {
            Log.v("takePictureLauncher", "Camera image uri: $cameraImageUri")
        }
    }

    private val requestPermissionLauncher =
        registerForActivityResult(ActivityResultContracts.RequestMultiplePermissions()) { permissions ->
            permissions.entries.forEach {
                Log.d("Permissions", "${it.key} = ${it.value}")
            }
        }

    @RequiresApi(Build.VERSION_CODES.TIRAMISU)
    @ExperimentalMaterial3Api
    override fun onCreate(savedInstanceState: Bundle?) {
        super.onCreate(savedInstanceState)

        // Keep screen on and maintain foreground-activity OOM priority (oom_score_adj -800).
        // FLAG_KEEP_SCREEN_ON prevents screen timeout.
        // setShowWhenLocked + setTurnScreenOn + requestDismissKeyguard ensure the activity
        // stays resumed (not paused) even when Samsung's security lock timer fires — a paused
        // activity loses foreground priority and becomes killable by lmkd at oom_score_adj 200.
        window.addFlags(WindowManager.LayoutParams.FLAG_KEEP_SCREEN_ON)
        setShowWhenLocked(true)
        setTurnScreenOn(true)
        (getSystemService(Context.KEYGUARD_SERVICE) as KeyguardManager)
            .requestDismissKeyguard(this, null)

        Log.i("MainActivity", "Starting benchmark service...")

        // Start the benchmark service
        startForegroundService(
            Intent(this, BenchmarkService::class.java)
        )

        // Poll for model ready, then run prompt
        val handler = Handler(Looper.getMainLooper())
        val checkModelTask = object : Runnable {
            var attempts = 0
            override fun run() {
                attempts++
                Log.i("MainActivity", "Checking if model is loaded (attempt $attempts)...")

                try {
                    if (BenchmarkService.isInitialized() && BenchmarkService.INSTANCE.isModelLoaded()) {
                        Log.i("MainActivity", "Model loaded! Reading prompt from file...")

                        // Try multiple locations for prompt file
                        val promptLocations = listOf(
                            java.io.File("/sdcard/Android/data/ai.mlc.mlcchat/files/prompt.txt"),
                            java.io.File("/sdcard/Download/prompt.txt"),
                            java.io.File("/sdcard/prompt.txt"),
                            java.io.File(getExternalFilesDir(null), "prompt.txt")
                        )

                        var promptFile: java.io.File? = null
                        for (location in promptLocations) {
                            if (location.exists() && location.canRead()) {
                                promptFile = location
                                Log.i("MainActivity", "Found prompt at: ${location.absolutePath}")
                                break
                            }
                        }

                        val prompt = if (promptFile != null) {
                            try {
                                promptFile.readText(Charsets.UTF_8).trim()
                            } catch (e: Exception) {
                                Log.e("MainActivity", "Failed to read prompt: ${e.message}")
                                "Explain how transformers work in machine learning."
                            }
                        } else {
                            Log.w("MainActivity", "No prompt file found, using default")
                            "Explain how transformers work in machine learning."
                        }

                        Log.i("MainActivity", "Running prompt: $prompt")
                        BenchmarkService.INSTANCE.runBenchmark(prompt)
                        Log.i("MainActivity", "Benchmark started, keeping activity alive to maintain foreground priority")
                    } else if (attempts < 60) {
                        // Try again in 500ms (max 30 seconds total)
                        handler.postDelayed(this, 500)
                    } else {
                        Log.e("MainActivity", "Timeout waiting for model to load after 30 seconds!")
                        finish()
                    }
                } catch (e: Exception) {
                    Log.e("MainActivity", "Error in check loop: ${e.message}", e)
                    if (attempts < 60) {
                        handler.postDelayed(this, 500)
                    } else {
                        finish()
                    }
                }
            }
        }

        // Start checking after 1 second
        handler.postDelayed(checkModelTask, 1000)
    }

    private fun requestNeededPermissions() {
        val permissionsToRequest = mutableListOf<String>()

        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.TIRAMISU) {
            if (ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.READ_MEDIA_IMAGES
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                permissionsToRequest.add(Manifest.permission.READ_MEDIA_IMAGES)
            }
            if (ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.CAMERA
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                permissionsToRequest.add(Manifest.permission.CAMERA)
            }
        } else {
            if (ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.READ_EXTERNAL_STORAGE
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                permissionsToRequest.add(Manifest.permission.READ_EXTERNAL_STORAGE)
            }
            if (ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.WRITE_EXTERNAL_STORAGE
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                permissionsToRequest.add(Manifest.permission.WRITE_EXTERNAL_STORAGE)
            }
            if (ContextCompat.checkSelfPermission(
                    this,
                    Manifest.permission.CAMERA
                ) != PackageManager.PERMISSION_GRANTED
            ) {
                permissionsToRequest.add(Manifest.permission.CAMERA)
            }
        }

        if (permissionsToRequest.isNotEmpty()) {
            requestPermissionLauncher.launch(permissionsToRequest.toTypedArray())
        }
    }

    fun pickImageFromGallery() {
        pickImageLauncher.launch("image/*")
    }

    fun takePhoto() {
        val contentValues = ContentValues().apply {
            val timeFormatter = SimpleDateFormat("yyyyMMdd_HHmmss", Locale.getDefault())
            val fileName = "IMG_${timeFormatter.format(Date())}.jpg"
            put(MediaStore.Images.Media.DISPLAY_NAME, fileName)
            put(MediaStore.Images.Media.MIME_TYPE, "image/jpeg")
            put(MediaStore.Images.Media.DATE_ADDED, System.currentTimeMillis() / 1000)
        }

        cameraImageUri = contentResolver.insert(
            MediaStore.Images.Media.EXTERNAL_CONTENT_URI,
            contentValues
        )

        takePictureLauncher.launch(cameraImageUri)
    }
}