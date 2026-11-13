package com.echomebetter.engine.manifest

import kotlinx.serialization.json.JsonArray
import kotlinx.serialization.json.JsonElement
import kotlinx.serialization.json.JsonNull
import kotlinx.serialization.json.JsonObject
import kotlinx.serialization.json.JsonPrimitive
import kotlinx.serialization.json.booleanOrNull
import kotlinx.serialization.json.doubleOrNull
import kotlinx.serialization.json.longOrNull

/** Typed reads from a JSON tree, failing with [ManifestException] and the path of the offending value. */
internal object Json {
    fun obj(value: JsonElement?, where: String): JsonObject =
        value as? JsonObject ?: throw ManifestException("$where must be an object")

    fun string(value: JsonElement?, where: String): String {
        val primitive = value as? JsonPrimitive
        if (primitive == null || !primitive.isString || primitive.content.isEmpty()) throw ManifestException("$where must be a non-empty string")
        return primitive.content
    }

    fun int(value: JsonElement?, where: String, min: Int): Int {
        val primitive = value as? JsonPrimitive
        val number = primitive?.takeUnless { it.isString }?.longOrNull
        if (number == null || number < min || number > Int.MAX_VALUE) throw ManifestException("$where must be an integer >= $min")
        return number.toInt()
    }

    fun long(value: JsonElement?, where: String, min: Long): Long {
        val number = (value as? JsonPrimitive)?.takeUnless { it.isString }?.longOrNull
        if (number == null || number < min) throw ManifestException("$where must be an integer >= $min")
        return number
    }

    fun double(value: JsonElement?, where: String): Double =
        (value as? JsonPrimitive)?.takeUnless { it.isString }?.doubleOrNull ?: throw ManifestException("$where must be a number")

    fun boolean(value: JsonElement?, where: String): Boolean =
        (value as? JsonPrimitive)?.takeUnless { it.isString }?.booleanOrNull ?: throw ManifestException("$where must be true or false")

    fun array(value: JsonElement?, where: String): JsonArray =
        value as? JsonArray ?: throw ManifestException("$where must be a list")

    /** The value, or null when it is absent or JSON null. */
    fun optional(value: JsonElement?): JsonElement? = value.takeUnless { it == null || it is JsonNull }
}
