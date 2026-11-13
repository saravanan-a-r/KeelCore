package com.echomebetter.rewrite.settings

import android.content.Context
import com.echomebetter.engine.Style
import kotlinx.coroutines.flow.MutableStateFlow
import kotlinx.coroutines.flow.StateFlow
import kotlinx.coroutines.flow.asStateFlow

/** The few choices the user makes, kept in one preferences file. */
class RewriteSettings(context: Context) {
    private val preferences = context.getSharedPreferences("rewrite", Context.MODE_PRIVATE)

    private val _style = MutableStateFlow(preferences.getString(KEY_STYLE, null)?.let(Style::fromId) ?: Style.GRAMMAR)

    /** The style the Rewrite button uses. */
    val style: StateFlow<Style> = _style.asStateFlow()

    fun selectStyle(style: Style) {
        preferences.edit().putString(KEY_STYLE, style.id).apply()
        _style.value = style
    }

    private val _licenseAccepted = MutableStateFlow(preferences.getBoolean(KEY_LICENSE, false))

    /** The user agreed to the Thelivu License before the first download. */
    val licenseAccepted: StateFlow<Boolean> = _licenseAccepted.asStateFlow()

    fun setLicenseAccepted(accepted: Boolean) {
        preferences.edit().putBoolean(KEY_LICENSE, accepted).apply()
        _licenseAccepted.value = accepted
    }

    private companion object {
        const val KEY_STYLE = "style"
        const val KEY_LICENSE = "license_accepted"
    }
}
