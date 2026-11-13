package com.echomebetter.rewrite.engine

import com.echomebetter.engine.RewriteEngine
import com.echomebetter.engine.RewriteListener
import com.echomebetter.engine.RewriteResult
import com.echomebetter.engine.Style
import com.echomebetter.engine.onnx.CrossKvRunner
import com.echomebetter.engine.onnx.LoraAdapterPair
import com.echomebetter.engine.onnx.OnnxSessions
import com.echomebetter.engine.onnx.SessionSettings
import com.echomebetter.engine.tokenizer.Tokenizer
import com.echomebetter.rewrite.model.InstalledState
import com.echomebetter.rewrite.model.ModelRepository
import com.echomebetter.modelstore.InstalledModel
import kotlinx.coroutines.CoroutineScope
import kotlinx.coroutines.Job
import kotlinx.coroutines.asCoroutineDispatcher
import kotlinx.coroutines.delay
import kotlinx.coroutines.flow.drop
import kotlinx.coroutines.launch
import kotlinx.coroutines.sync.Mutex
import kotlinx.coroutines.sync.withLock
import kotlinx.coroutines.withContext
import java.util.concurrent.Executors

/**
 * Keeps the model in memory while it is being used, and only then.
 *
 * The base model (about 0.8 GB) is loaded on the first rewrite, with the adapter of the style asked for; a different
 * style swaps the adapter (about 70 MB) and keeps the base. After [idleUnloadMs] without a rewrite, when the system
 * asks for memory, or when the installed model changes, everything is released. All model work runs on one
 * background thread, one rewrite at a time.
 */
class EngineHost(
    private val models: ModelRepository,
    private val scope: CoroutineScope,
    private val idleUnloadMs: Long = 3 * 60_000L,
) : Rewriter {
    private class Loaded(val model: InstalledModel, val sessions: OnnxSessions, val engine: RewriteEngine, val runner: CrossKvRunner)

    private class ActiveAdapter(val style: Style, val pair: LoraAdapterPair, val runner: CrossKvRunner)

    private val dispatcher = Executors.newSingleThreadExecutor { task ->
        Thread(task, "thelivu-engine").apply { priority = Thread.NORM_PRIORITY - 1 }
    }.asCoroutineDispatcher()
    private val lock = Mutex()
    private var loaded: Loaded? = null
    private var adapter: ActiveAdapter? = null
    private var idleJob: Job? = null

    init {
        // A model installed, replaced or removed: what is in memory may no longer match the files.
        scope.launch { models.installed.drop(1).collect { release() } }
    }

    override suspend fun rewrite(style: Style, text: String, onPhase: (RewritePhase) -> Unit): RewriteResult = withContext(dispatcher) {
        lock.withLock {
            // A pending idle unload waits for this lock; cancelling it here means it never runs after this rewrite.
            idleJob?.cancel()
            try {
                val runner = runnerFor(style, onPhase)
                val engine = loaded!!.engine
                val styleToken = loaded!!.model.manifest.adapters.getValue(style).styleToken
                engine.rewrite(style, styleToken, runner, text, object : RewriteListener {
                    override fun onToken(count: Int, readText: () -> String) = onPhase(RewritePhase.Writing(readText))
                })
            } finally {
                idleJob = scope.launch {
                    delay(idleUnloadMs)
                    release()
                }
            }
        }
    }

    /** Frees the model's memory now; the next rewrite loads it again. */
    suspend fun release() = withContext(dispatcher) {
        lock.withLock { unload() }
    }

    private fun runnerFor(style: Style, onPhase: (RewritePhase) -> Unit): CrossKvRunner {
        val current = loaded ?: load(onPhase)
        adapter?.let { if (it.style == style) return it.runner }
        val (encoderFile, decoderFile) = current.model.adapterFiles(style) ?: throw StyleNotInstalledException(style)
        adapter?.pair?.close()
        adapter = null
        val pair = try {
            LoraAdapterPair.load(encoderFile, decoderFile)
        } catch (error: Exception) {
            throw ModelUnavailableException("could not load the ${style.id} adapter", error)
        }
        return current.runner.withAdapter(pair).also { adapter = ActiveAdapter(style, pair, it) }
    }

    private fun load(onPhase: (RewritePhase) -> Unit): Loaded {
        val model = (models.installed.value as? InstalledState.Ready)?.model ?: throw ModelUnavailableException("the model is not installed")
        onPhase(RewritePhase.Loading)
        val sessions = try {
            OnnxSessions.open(model.encoder, model.decoder, SessionSettings(intraOpThreads = threads()))
        } catch (error: Exception) {
            throw ModelUnavailableException("could not open the model", error)
        } catch (error: OutOfMemoryError) {
            throw ModelUnavailableException("not enough memory for the model", error)
        }
        try {
            val tokenizer = Tokenizer.fromJson(model.tokenizer.readText())
            val runner = CrossKvRunner(sessions.env, sessions.encoder, sessions.decoder, model.index.base.architecture)
            return Loaded(model, sessions, RewriteEngine(model.index.base, tokenizer), runner).also { loaded = it }
        } catch (error: Exception) {
            sessions.close()
            throw ModelUnavailableException("could not prepare the model", error)
        }
    }

    private fun unload() {
        adapter?.pair?.close()
        adapter = null
        loaded?.sessions?.close()
        loaded = null
    }

    private companion object {
        /** Half the cores, two to four: the fast cores do the work, and the phone stays responsive. */
        fun threads(): Int = (Runtime.getRuntime().availableProcessors() / 2).coerceIn(2, 4)
    }
}
