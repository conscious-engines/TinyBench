package ai.mlc.mlcchat

import android.content.BroadcastReceiver
import android.content.Context
import android.content.Intent
import ai.mlc.mlcllm.MLCEngine

class PromptReceiver : BroadcastReceiver() {
    override fun onReceive(context: Context, intent: Intent) {
        val prompt = intent.getStringExtra("prompt") ?: return
        BenchmarkService.INSTANCE.runBenchmark(prompt)

    }
}

