package com.echomebetter.rewrite.engine

import com.echomebetter.engine.RewriteResult
import com.echomebetter.engine.Style

/** What the keyboard shows while a rewrite runs. */
sealed interface RewritePhase {
    /** The model is being read into memory (the first rewrite after a while). */
    data object Loading : RewritePhase

    /** The model is writing; [readText] decodes the rewrite so far. */
    class Writing(val readText: () -> String) : RewritePhase
}

/** Rewrites text in a style. Implemented by [EngineHost]; faked in tests. */
interface Rewriter {
    /** [onPhase] is called on the rewriting thread; a caller showing it moves to its own thread. */
    suspend fun rewrite(style: Style, text: String, onPhase: (RewritePhase) -> Unit): RewriteResult
}

/** The style's adapter is not on the phone. */
class StyleNotInstalledException(val style: Style) : Exception("${style.id} is not installed")

/** The model could not be loaded (not installed, damaged, or not enough memory). */
class ModelUnavailableException(message: String, cause: Throwable? = null) : Exception(message, cause)
