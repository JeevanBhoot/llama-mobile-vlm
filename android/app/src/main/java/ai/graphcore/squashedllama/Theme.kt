// Copyright (c) 2026 Graphcore Ltd. All rights reserved.

package ai.graphcore.squashedllama

import androidx.compose.foundation.isSystemInDarkTheme
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Typography
import androidx.compose.material3.dynamicDarkColorScheme
import androidx.compose.material3.dynamicLightColorScheme
import androidx.compose.runtime.Composable
import androidx.compose.ui.platform.LocalContext
import androidx.compose.ui.text.TextStyle
import androidx.compose.ui.text.font.FontFamily
import androidx.compose.ui.text.font.FontWeight
import androidx.compose.ui.unit.TextUnit
import androidx.compose.ui.unit.sp

private const val LargeFontScale = 1.5f

private fun TextUnit.scaled(scale: Float): TextUnit =
    if (this == TextUnit.Unspecified) this else this * scale

private fun TextStyle.scaled(scale: Float): TextStyle =
    copy(
        fontSize = fontSize.scaled(scale),
        lineHeight = lineHeight.scaled(scale)
    )

private fun Typography.scaled(scale: Float): Typography =
    copy(
        displayLarge = displayLarge.scaled(scale),
        displayMedium = displayMedium.scaled(scale),
        displaySmall = displaySmall.scaled(scale),
        headlineLarge = headlineLarge.scaled(scale),
        headlineMedium = headlineMedium.scaled(scale),
        headlineSmall = headlineSmall.scaled(scale),
        titleLarge = titleLarge.scaled(scale),
        titleMedium = titleMedium.scaled(scale),
        titleSmall = titleSmall.scaled(scale),
        bodyLarge = bodyLarge.scaled(scale),
        bodyMedium = bodyMedium.scaled(scale),
        bodySmall = bodySmall.scaled(scale),
        labelLarge = labelLarge.scaled(scale),
        labelMedium = labelMedium.scaled(scale),
        labelSmall = labelSmall.scaled(scale),
    )

@Composable
fun CustomTheme(
    darkTheme: Boolean = isSystemInDarkTheme(),
    largeFonts: Boolean = false,
    content: @Composable () -> Unit
) {
    val context = LocalContext.current
    val typography = Typography(
        bodyLarge = TextStyle(
            fontFamily = FontFamily.Default,
            fontWeight = FontWeight.Normal,
            fontSize = 16.sp,
            lineHeight = 24.sp,
            letterSpacing = 0.5.sp
        )
    ).scaled(if (largeFonts) LargeFontScale else 1f)

    MaterialTheme(
        colorScheme =
        if (darkTheme) dynamicDarkColorScheme(context)
        else dynamicLightColorScheme(context),
        typography = typography,
        content = content
    )
}
