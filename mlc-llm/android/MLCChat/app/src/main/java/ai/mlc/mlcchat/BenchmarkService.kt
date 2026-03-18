package ai.mlc.mlcchat

import android.app.Notification
import android.app.NotificationChannel
import android.app.NotificationManager
import android.app.Service
import android.content.Context
import android.content.Intent
import android.content.IntentFilter
import android.os.BatteryManager
import android.os.Build
import android.os.IBinder
import android.os.PowerManager
import android.util.Log
import ai.mlc.mlcllm.MLCEngine
import ai.mlc.mlcllm.OpenAIProtocol
import kotlinx.coroutines.*
import java.io.File
import java.text.SimpleDateFormat
import java.util.*

class BenchmarkService : Service() {

    companion object {
        private const val TAG = "BENCH"
        private const val CHANNEL_ID = "bench"
        private const val WARMUP_ITERATIONS = 1
        private const val BENCHMARK_ITERATIONS = 20
        private const val INTER_ITERATION_DELAY_MS = 1000L
        private var instance: BenchmarkService? = null

        val INSTANCE: BenchmarkService
            get() = instance ?: throw IllegalStateException("Service not initialized")

        fun isInitialized() = instance != null
    }

    private lateinit var engine: MLCEngine
    private val scope = CoroutineScope(Dispatchers.IO + SupervisorJob())
    private var modelLoaded = false
    private var wakeLock: PowerManager.WakeLock? = null

    // -------------------------------------------------------------------------
    // Lifecycle
    // -------------------------------------------------------------------------

    override fun onCreate() {
        super.onCreate()
        instance = this
        createNotificationChannel()
        startForeground(1, createNotification())

        val pm = getSystemService(Context.POWER_SERVICE) as PowerManager
        wakeLock = pm.newWakeLock(PowerManager.PARTIAL_WAKE_LOCK, "MLCBenchmark:WakeLock").also {
            it.acquire(60 * 60 * 1000L) // 60 minutes max
        }

        scope.launch {
            try {
                initializeModel()
            } catch (e: Exception) {
                Log.e(TAG, "Failed to initialize model", e)
            }
        }
    }

    override fun onStartCommand(intent: Intent?, flags: Int, startId: Int): Int {
        return START_NOT_STICKY
    }

    override fun onBind(intent: Intent?): IBinder? = null

    override fun onDestroy() {
        instance = null
        scope.cancel()
        if (::engine.isInitialized) {
            try { engine.unload() } catch (_: Exception) {}
        }
        wakeLock?.let { if (it.isHeld) it.release() }
        super.onDestroy()
    }

    // -------------------------------------------------------------------------
    // Model init
    // -------------------------------------------------------------------------

    private suspend fun initializeModel() = withContext(Dispatchers.IO) {
        Log.i(TAG, "============================================")
        Log.i(TAG, "INITIALIZING MODEL")
        Log.i(TAG, "============================================")

        val modelDir = File(filesDir, "qwen2.5-1.5b-android-opencl-complete")
        if (!modelDir.exists() || !File(modelDir, "mlc-chat-config.json").exists()) {
            Log.i(TAG, "Copying model from assets to internal storage...")
            val filesToCopy = listOf(
                "lib", "params", "mlc-chat-config.json", "merges.txt",
                "tokenizer.json", "tokenizer_config.json", "vocab.json", "tensor-cache.json"
            )
            modelDir.mkdirs()
            for (item in filesToCopy) {
                copyAssetFolder(item, File(modelDir, item).absolutePath)
            }
            Log.i(TAG, "Model copied successfully")
        } else {
            Log.i(TAG, "Model already exists in internal storage")
        }

        val configFile = File(modelDir, "mlc-chat-config.json")
        if (!configFile.exists()) {
            Log.e(TAG, "mlc-chat-config.json NOT FOUND at: ${configFile.absolutePath}")
            return@withContext
        }

        Log.i(TAG, "Creating MLCEngine...")
        engine = MLCEngine()

        Log.i(TAG, "Loading model into engine...")
        val reloadStart = System.currentTimeMillis()
        try {
            engine.reload(modelDir.absolutePath, "libmodel_android")
            Log.i(TAG, "Model loaded in ${System.currentTimeMillis() - reloadStart}ms")
            Log.i(TAG, "============================================")
            modelLoaded = true
        } catch (e: Exception) {
            Log.e(TAG, "Failed to load model into engine", e)
        }
    }

    private fun copyAssetFolder(assetPath: String, targetPath: String) {
        val assetManager = assets
        val subFiles = try { assetManager.list(assetPath) } catch (e: Exception) { null }
        if (subFiles != null && subFiles.isNotEmpty()) {
            File(targetPath).mkdirs()
            for (file in subFiles) {
                copyAssetFolder("$assetPath/$file", "$targetPath/$file")
            }
        } else {
            assetManager.open(assetPath).use { input ->
                File(targetPath).outputStream().use { output -> input.copyTo(output) }
            }
        }
    }

    fun isModelLoaded() = modelLoaded

    // -------------------------------------------------------------------------
    // Public entry point: runs warmup + benchmark iterations
    // -------------------------------------------------------------------------

    fun runBenchmark(prompt: String) {
        if (!modelLoaded) {
            Log.e(TAG, "Model not loaded yet!")
            return
        }

        scope.launch {
            Log.i(TAG, "############################################")
            Log.i(TAG, "BENCHMARK START  prompt=\"${prompt.take(80)}\"")
            Log.i(TAG, "  warmup=$WARMUP_ITERATIONS  iterations=$BENCHMARK_ITERATIONS")
            Log.i(TAG, "############################################")

            repeat(WARMUP_ITERATIONS) { idx ->
                Log.i(TAG, "--- WARMUP ${idx + 1}/$WARMUP_ITERATIONS ---")
                runSingleInference(prompt, logResult = false)
                delay(INTER_ITERATION_DELAY_MS)
            }

            Log.i(TAG, "--- WARMUP COMPLETE, starting timed iterations ---")

            val csvFile = prepareCsvFile()
            repeat(BENCHMARK_ITERATIONS) { idx ->
                val iterNum = idx + 1
                Log.i(TAG, "--- ITERATION $iterNum/$BENCHMARK_ITERATIONS ---")
                val result = runSingleInference(prompt, logResult = true)
                if (result != null) {
                    appendResultToCsv(csvFile, iterNum, prompt, result)
                }
                if (iterNum < BENCHMARK_ITERATIONS) {
                    delay(INTER_ITERATION_DELAY_MS)
                }
            }

            Log.i(TAG, "############################################")
            Log.i(TAG, "BENCHMARK COMPLETE  results -> ${csvFile.absolutePath}")
            Log.i(TAG, "############################################")
        }
    }

    // -------------------------------------------------------------------------
    // Single inference
    // -------------------------------------------------------------------------

    private data class IterationResult(
        val timestamp: Long,
        val totalTokens: Int,
        val prefillTimeMs: Long,
        val decodeTokens: Int,
        val decodeTimeMs: Long,
        val totalTimeMs: Long,
        val decodeThroughput: Double,
        val initialBatteryLevel: Int,
        val finalBatteryLevel: Int,
        val initialBatteryTemp: Float,
        val finalBatteryTemp: Float,
        val batteryHealth: String,
        val cpuTempEnd: Float,
        val gpuTempEnd: Float,
        val gpuFreq: Int,
        val gpuMaxFreq: Int,
        val maxBatteryTemp: Float,
        val maxCpuTemp: Float,
        val maxGpuTemp: Float,
        val initialPowerMw: Float,
        val finalPowerMw: Float,
        val avgPowerMw: Float,
        val peakPowerMw: Float,
        val energyPerTokenMj: Float
    )

    private suspend fun runSingleInference(prompt: String, logResult: Boolean): IterationResult? {
        engine.reset()
        return try {
            coroutineScope {
            val startTime = System.currentTimeMillis()
            val initialBatteryLevel = getBatteryLevel()
            val initialBatteryTemp = getBatteryTemperature()
            val initialBatteryPowerStat = getBatteryPowerStats()

            var maxBatteryTemp = initialBatteryTemp
            var maxCpuTemp = getCpuTemperature()
            var maxGpuTemp = getGpuTemperature()
            var peakPowerMw = initialBatteryPowerStat[2]
            var powerSampleSum = initialBatteryPowerStat[2]
            var powerSampleCount = 1

            val messages = listOf(
                OpenAIProtocol.ChatCompletionMessage(
                    role = OpenAIProtocol.ChatCompletionRole.user,
                    content = prompt
                )
            )

            val responseChannel = engine.chat.completions.create(messages = messages, max_tokens = 700, stream = true)

            var tokenCount = 0
            var firstTokenTime: Long? = null
            var lastTokenTime: Long? = null

            val monitorJob = launch(Dispatchers.IO) {
                while (isActive) {
                    try {
                        val bt = getBatteryTemperature()
                        val ct = getCpuTemperature()
                        val gt = getGpuTemperature()
                        val powerStats = getBatteryPowerStats()
                        val pw = powerStats[2]
                        if (bt > maxBatteryTemp) maxBatteryTemp = bt
                        if (ct > maxCpuTemp) maxCpuTemp = ct
                        if (gt > maxGpuTemp) maxGpuTemp = gt
                        if (pw > peakPowerMw) peakPowerMw = pw
                        powerSampleSum += pw
                        powerSampleCount++
                        delay(200)
                    } catch (_: Exception) {}
                }
            }

            try {
            for (chunk in responseChannel) {
                chunk.choices.firstOrNull()?.delta?.content?.let { content ->
                    val now = System.currentTimeMillis()
                    if (firstTokenTime == null) firstTokenTime = now
                    val textContent = content.toString().substringAfter("text=").substringBefore(",").trim()
                    if (textContent.isNotEmpty()) {
                        tokenCount++
                        lastTokenTime = now
                    }
                }
            }
            } finally {
                monitorJob.cancel()
            }

            val totalTime = System.currentTimeMillis() - startTime
            val prefillTime = firstTokenTime?.let { it - startTime } ?: 0L
            val decodeTime = if (firstTokenTime != null && lastTokenTime != null)
                lastTokenTime!! - firstTokenTime!! else 0L
            val decodeTokens = if (tokenCount > 0) tokenCount - 1 else 0
            val throughput = if (decodeTime > 0) decodeTokens * 1000.0 / decodeTime else 0.0

            val finalBatteryLevel = getBatteryLevel()
            val finalBatteryTemp = getBatteryTemperature()
            val batteryHealth = getBatteryHealth()
            val cpuTempEnd = getCpuTemperature()
            val gpuTempEnd = getGpuTemperature()
            val gpuFreq = getGpuFrequency()
            val gpuMaxFreq = getGpuMaxFrequency()
            val finalBatteryPowerStat = getBatteryPowerStats()

            val initialPower = initialBatteryPowerStat[2]
            val finalPower = finalBatteryPowerStat[2]
            val avgPower = powerSampleSum / powerSampleCount
            val energyPerToken = (avgPower * decodeTime) / (decodeTokens * 1000f)  // µJ → mJ

            val result = IterationResult(
                timestamp = System.currentTimeMillis(),
                totalTokens = tokenCount,
                prefillTimeMs = prefillTime,
                decodeTokens = decodeTokens,
                decodeTimeMs = decodeTime,
                totalTimeMs = totalTime,
                decodeThroughput = throughput,
                initialBatteryLevel = initialBatteryLevel,
                finalBatteryLevel = finalBatteryLevel,
                initialBatteryTemp = initialBatteryTemp,
                finalBatteryTemp = finalBatteryTemp,
                batteryHealth = batteryHealth,
                cpuTempEnd = cpuTempEnd,
                gpuTempEnd = gpuTempEnd,
                gpuFreq = gpuFreq,
                gpuMaxFreq = gpuMaxFreq,
                maxBatteryTemp = maxBatteryTemp,
                maxCpuTemp = maxCpuTemp,
                maxGpuTemp = maxGpuTemp,
                initialPowerMw = initialPower,
                finalPowerMw = finalPower,
                avgPowerMw = avgPower,
                peakPowerMw = peakPowerMw,
                energyPerTokenMj = energyPerToken
            )

            if (logResult) {
                Log.i(TAG, "  tokens=$tokenCount  prefill=${prefillTime}ms  decode=${decodeTime}ms  total=${totalTime}ms  tput=${String.format("%.3f", throughput)} tok/s")
                Log.i(TAG, "  batt: ${initialBatteryLevel}%→${finalBatteryLevel}%  temp: ${String.format("%.1f", initialBatteryTemp)}→${String.format("%.1f", finalBatteryTemp)}°C (max ${String.format("%.1f", maxBatteryTemp)}°C)")
                Log.i(TAG, "  cpu: ${String.format("%.1f", cpuTempEnd)}°C (max ${String.format("%.1f", maxCpuTemp)}°C)  gpu: ${String.format("%.1f", gpuTempEnd)}°C (max ${String.format("%.1f", maxGpuTemp)}°C)  freq: $gpuFreq/$gpuMaxFreq MHz")
                Log.i(TAG, "  power: avg=${String.format("%.1f", avgPower)}mW  peak=${String.format("%.1f", peakPowerMw)}mW  energy=${String.format("%.3f", energyPerToken)}mJ/tok")
            }

            result
            } // end coroutineScope
        } catch (e: CancellationException) {
            throw e
        } catch (e: Exception) {
            Log.e(TAG, "Error during inference", e)
            null
        }
    }

    // -------------------------------------------------------------------------
    // CSV helpers
    // -------------------------------------------------------------------------

    private fun prepareCsvFile(): File {
        val csvFile = File(getExternalFilesDir(null), "benchmark_log.csv")
        csvFile.writeText(
            "iteration,timestamp,date_time,prompt,total_tokens,prefill_ms,decode_tokens," +
                    "decode_ms,total_ms,decode_tok_s," +
                    "battery_start_%,battery_end_%,battery_delta_%," +
                    "temp_start_c,temp_end_c,temp_delta_c," +
                    "battery_health,cpu_temp_c,gpu_temp_c,gpu_freq_mhz,gpu_max_freq_mhz," +
                    "max_battery_temp_c,max_cpu_temp_c,max_gpu_temp_c," +
                    "initial_power_mw,final_power_mw,avg_power_mw,peak_power_mw,energy_per_token_mj\n",
            Charsets.UTF_8
        )
        return csvFile
    }

    private fun appendResultToCsv(csvFile: File, iteration: Int, prompt: String, r: IterationResult) {
        try {
            val dateFormat = SimpleDateFormat("yyyy-MM-dd HH:mm:ss", Locale.US)
            val batteryDelta = r.finalBatteryLevel - r.initialBatteryLevel
            val tempDelta = r.finalBatteryTemp - r.initialBatteryTemp
            val safePrompt = prompt.take(100).replace("\"", "\"\"")

            csvFile.appendText(
                "$iteration,${r.timestamp},${dateFormat.format(Date(r.timestamp))},\"$safePrompt\"," +
                        "${r.totalTokens},${r.prefillTimeMs},${r.decodeTokens},${r.decodeTimeMs},${r.totalTimeMs}," +
                        "${String.format("%.3f", r.decodeThroughput)}," +
                        "${r.initialBatteryLevel},${r.finalBatteryLevel},$batteryDelta," +
                        "${String.format("%.1f", r.initialBatteryTemp)},${String.format("%.1f", r.finalBatteryTemp)},${String.format("%.1f", tempDelta)}," +
                        "${r.batteryHealth},${String.format("%.1f", r.cpuTempEnd)},${String.format("%.1f", r.gpuTempEnd)},${r.gpuFreq},${r.gpuMaxFreq}," +
                        "${String.format("%.1f", r.maxBatteryTemp)},${String.format("%.1f", r.maxCpuTemp)},${String.format("%.1f", r.maxGpuTemp)}," +
                        "${String.format("%.1f", r.initialPowerMw)},${String.format("%.1f", r.finalPowerMw)}," +
                        "${String.format("%.1f", r.avgPowerMw)},${String.format("%.1f", r.peakPowerMw)}," +
                        "${String.format("%.3f", r.energyPerTokenMj)}\n",
                Charsets.UTF_8
            )
        } catch (e: Exception) {
            Log.e(TAG, "Failed to write CSV row", e)
        }
    }

    // -------------------------------------------------------------------------
    // Telemetry
    // -------------------------------------------------------------------------

    private fun getBatteryLevel(): Int {
        val s = registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        return s?.getIntExtra(BatteryManager.EXTRA_LEVEL, -1) ?: -1
    }

    private fun getBatteryTemperature(): Float {
        val s = registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val temp = s?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, -1) ?: -1
        return temp / 10.0f
    }

    private fun getBatteryHealth(): String {
        val s = registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        return when (s?.getIntExtra(BatteryManager.EXTRA_HEALTH, -1)) {
            BatteryManager.BATTERY_HEALTH_GOOD -> "Good"
            BatteryManager.BATTERY_HEALTH_OVERHEAT -> "Overheat"
            BatteryManager.BATTERY_HEALTH_DEAD -> "Dead"
            BatteryManager.BATTERY_HEALTH_OVER_VOLTAGE -> "Over Voltage"
            BatteryManager.BATTERY_HEALTH_COLD -> "Cold"
            else -> "Unknown"
        }
    }

    private fun getBatteryPowerStats(): FloatArray {
        // Returns [voltage_mV, current_mA, power_mW, temperature_C]
        val intent = registerReceiver(null, IntentFilter(Intent.ACTION_BATTERY_CHANGED))
        val voltage = intent?.getIntExtra(BatteryManager.EXTRA_VOLTAGE, 0)?.toFloat() ?: 0f
        val temperature = (intent?.getIntExtra(BatteryManager.EXTRA_TEMPERATURE, 0) ?: 0) / 10f
        val bm = getSystemService(BATTERY_SERVICE) as BatteryManager

        val rawCurrent = bm.getIntProperty(BatteryManager.BATTERY_PROPERTY_CURRENT_NOW)
//        Log.d(TAG, "  [power_debug] voltage=${voltage}mV  raw_current=${rawCurrent}  " +
//                "BATTERY_PROPERTY_SUPPORTED=${rawCurrent != Int.MIN_VALUE}")

        // BATTERY_PROPERTY_CURRENT_NOW is µA on most devices, not mA
        val currentMa = Math.abs(rawCurrent).toFloat()  // µA -> mA
        val power = (voltage / 1000f) * Math.abs(currentMa)  // V * mA = mW
//        Log.d(TAG, "  [power_debug] current_ma=${currentMa}mA  power=${power}mW")

        return floatArrayOf(voltage, currentMa, power, temperature)
    }

    private fun getCpuTemperature(): Float {
        return try {
            File("/sys/class/thermal/thermal_zone0/temp").readText(Charsets.UTF_8).trim().toFloat() / 1000
        } catch (e: Exception) { getBatteryTemperature() }
    }

    private fun getGpuTemperature(): Float {
        return try {
            File("/sys/class/kgsl/kgsl-3d0/temp").readText(Charsets.UTF_8).trim().toFloat() / 1000
        } catch (e: Exception) { 0f }
    }

    private fun getGpuFrequency(): Int {
        return try {
            File("/sys/class/kgsl/kgsl-3d0/devfreq/cur_freq").readText(Charsets.UTF_8).trim().toInt() / 1000000
        } catch (e: Exception) { 0 }
    }

    private fun getGpuMaxFrequency(): Int {
        return try {
            File("/sys/class/kgsl/kgsl-3d0/devfreq/max_freq").readText(Charsets.UTF_8).trim().toInt() / 1000000
        } catch (e: Exception) { 0 }
    }

    // -------------------------------------------------------------------------
    // Notification
    // -------------------------------------------------------------------------

    private fun createNotificationChannel() {
        if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            getSystemService(NotificationManager::class.java).createNotificationChannel(
                NotificationChannel(CHANNEL_ID, "Benchmark Service", NotificationManager.IMPORTANCE_LOW)
            )
        }
    }

    private fun createNotification(): Notification {
        return if (Build.VERSION.SDK_INT >= Build.VERSION_CODES.O) {
            Notification.Builder(this, CHANNEL_ID)
                .setContentTitle("MLC Benchmark").setContentText("Running...")
                .setSmallIcon(android.R.drawable.ic_dialog_info).build()
        } else {
            @Suppress("DEPRECATION")
            Notification.Builder(this)
                .setContentTitle("MLC Benchmark").setContentText("Running...")
                .setSmallIcon(android.R.drawable.ic_dialog_info).build()
        }
    }
}