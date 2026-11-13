package com.echomebetter.rewrite.model

import com.echomebetter.engine.Style

/** Something to download: the base model, or one style's adapter. */
sealed interface InstallTarget {
    /** Stable text form, used as the work's input and tag. */
    val key: String

    data object Base : InstallTarget {
        override val key = "base"
    }

    data class Adapter(val style: Style) : InstallTarget {
        override val key = "adapter:${style.id}"
    }

    companion object {
        fun parse(key: String?): InstallTarget? = when {
            key == Base.key -> Base
            key != null && key.startsWith("adapter:") -> Style.fromId(key.removePrefix("adapter:"))?.let(::Adapter)
            else -> null
        }
    }
}

/** Why a download stopped, as the setup screen explains it. */
enum class FailureReason { NETWORK, INTEGRITY, STORAGE, BASE_CHANGED, NO_ADAPTER, UNKNOWN }

/** A download that is queued, running or failed. A finished one shows as installed instead. */
sealed interface DownloadState {
    data class Queued(val waitingForWifi: Boolean) : DownloadState

    data class Running(val doneBytes: Long, val totalBytes: Long) : DownloadState

    data class Failed(val reason: FailureReason, val neededBytes: Long) : DownloadState
}

/** What is on the phone. */
sealed interface InstalledState {
    /** Not read yet. */
    data object Checking : InstalledState

    data object None : InstalledState

    data class Ready(val model: com.echomebetter.modelstore.InstalledModel) : InstalledState
}
