package com.echomebetter.engine.onnx

import ai.onnxruntime.OrtEnvironment
import ai.onnxruntime.OrtLoggingLevel
import ai.onnxruntime.OrtSession
import java.io.File

/** How the base model's sessions run. */
data class SessionSettings(
    /** Threads per operator; the phone's big cores are the useful ones, so this stays small. */
    val intraOpThreads: Int = 4,
)

/** The base model's two graphs, open. Closing them frees the model's memory. */
class OnnxSessions private constructor(
    val env: OrtEnvironment,
    val encoder: OrtSession,
    val decoder: OrtSession,
) : AutoCloseable {
    override fun close() {
        encoder.close()
        decoder.close()
    }

    companion object {
        fun open(encoderFile: File, decoderFile: File, settings: SessionSettings = SessionSettings()): OnnxSessions {
            val env = OrtEnvironment.getEnvironment()
            // No usage telemetry from onnxruntime: rewriting must not contact anything (the app also removes the
            // Android build's telemetry initializer from its manifest).
            env.setTelemetry(false)
            OrtSession.SessionOptions().use { options ->
                options.setOptimizationLevel(OrtSession.SessionOptions.OptLevel.ALL_OPT)
                options.setExecutionMode(OrtSession.SessionOptions.ExecutionMode.SEQUENTIAL)
                options.setIntraOpNumThreads(settings.intraOpThreads)
                options.setInterOpNumThreads(1)
                // The base graphs take their LoRA weights as overridable inputs, which onnxruntime warns about once per
                // weight on every load. That is the design, not a fault: only errors are worth logging.
                options.setSessionLogLevel(OrtLoggingLevel.ORT_LOGGING_LEVEL_ERROR)
                val encoder = env.createSession(encoderFile.path, options)
                try {
                    return OnnxSessions(env, encoder, env.createSession(decoderFile.path, options))
                } catch (error: Throwable) {
                    encoder.close()
                    throw error
                }
            }
        }
    }
}
