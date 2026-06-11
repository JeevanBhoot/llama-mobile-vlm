// Copyright (c) 2026 Graphcore Ltd. All rights reserved.

package ai.graphcore.llamamobiledemo

import androidx.activity.compose.BackHandler
import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.foundation.rememberScrollState
import androidx.compose.foundation.verticalScroll
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.material3.TextButton
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.text.font.FontStyle
import androidx.compose.ui.unit.dp

@Composable
fun AboutScreen(
    onBack: () -> Unit,
    modifier: Modifier = Modifier
) {
    BackHandler(onBack = onBack)

    Surface(modifier = modifier, color = MaterialTheme.colorScheme.surface) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .statusBarsPadding()
                .verticalScroll(rememberScrollState())
                .padding(24.dp),
            verticalArrangement = Arrangement.spacedBy(16.dp)
        ) {
            Text("Llama Mobile Demo (Graphcore)", style = MaterialTheme.typography.headlineSmall)

            Text(
                "Llama Mobile Demo is a demonstration app. It is provided for research and " +
                    "evaluation purposes only. Model outputs may be inaccurate, incomplete, or " +
                    "inappropriate and should not be relied on for production, safety-critical, " +
                    "medical, legal, or financial decisions.",
                style = MaterialTheme.typography.bodyMedium
            )
            Text(
                "This demo app does not receive security updates.",
                style = MaterialTheme.typography.bodyMedium
            )

            Text("Built with Llama.", style = MaterialTheme.typography.bodyMedium)
            Text(
                "This app is not affiliated with, sponsored by, or endorsed by Meta.",
                style = MaterialTheme.typography.bodyMedium
            )

            Text("Links", style = MaterialTheme.typography.titleMedium)
            Text("Blog post: coming soon", style = MaterialTheme.typography.bodyMedium)
            Text("Paper: coming soon", style = MaterialTheme.typography.bodyMedium)

            Text("Model Notice", style = MaterialTheme.typography.titleMedium)
            Text(
                "Llama 3.2 is licensed under the Llama 3.2 Community License, " +
                    "Copyright (c) Meta Platforms, Inc. All Rights Reserved.",
                style = MaterialTheme.typography.bodyMedium
            )

            Text("Open Source Notices", style = MaterialTheme.typography.titleMedium)
            Text(
                "This app uses AndroidX, Jetpack Compose, CameraX, Kotlin, and Android platform " +
                    "libraries. The native runtime uses nlohmann/json, cxxopts, and stb. " +
                    "See the project source and dependency license files for full notices.",
                style = MaterialTheme.typography.bodyMedium
            )

            Text(
                "Build date: ${BuildConfig.BUILD_DATE}",
                style = MaterialTheme.typography.bodySmall.copy(fontStyle = FontStyle.Italic)
            )

            TextButton(onClick = onBack, modifier = Modifier.align(Alignment.End)) {
                Text("Back")
            }
        }
    }
}
