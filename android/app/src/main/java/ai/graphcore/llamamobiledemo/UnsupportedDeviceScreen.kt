// Copyright (c) 2026 Graphcore Ltd. All rights reserved.

package ai.graphcore.llamamobiledemo

import androidx.compose.foundation.layout.Arrangement
import androidx.compose.foundation.layout.Column
import androidx.compose.foundation.layout.Spacer
import androidx.compose.foundation.layout.fillMaxSize
import androidx.compose.foundation.layout.height
import androidx.compose.foundation.layout.padding
import androidx.compose.foundation.layout.statusBarsPadding
import androidx.compose.material3.Button
import androidx.compose.material3.MaterialTheme
import androidx.compose.material3.Surface
import androidx.compose.material3.Text
import androidx.compose.runtime.Composable
import androidx.compose.ui.Alignment
import androidx.compose.ui.Modifier
import androidx.compose.ui.tooling.preview.Preview
import androidx.compose.ui.unit.dp

@Composable
fun UnsupportedDeviceScreen(
    reason: String,
    modifier: Modifier = Modifier,
    onOk: () -> Unit = {}
) {
    Surface(modifier = modifier, color = MaterialTheme.colorScheme.background) {
        Column(
            modifier = Modifier
                .fillMaxSize()
                .statusBarsPadding()
                .padding(24.dp),
            verticalArrangement = Arrangement.Center,
            horizontalAlignment = Alignment.Start
        ) {
            Text(
                text = "Unsupported device",
                style = MaterialTheme.typography.headlineSmall
            )
            Spacer(modifier = Modifier.height(12.dp))
            Text(
                text = "Sorry, this demo requires newer ARM CPU extensions for the included models." +
                        "This device cannot run the native inference engine.",
                style = MaterialTheme.typography.bodyLarge
            )
            Spacer(modifier = Modifier.height(12.dp))
            Text(
                text = reason,
                style = MaterialTheme.typography.bodyMedium,
                color = MaterialTheme.colorScheme.onSurfaceVariant
            )
            Spacer(modifier = Modifier.height(24.dp))
            Button(onClick = onOk) {
                Text("Close")
            }
        }
    }
}

@Preview(showBackground = true)
@Composable
private fun UnsupportedDeviceScreenPreview() {
    CustomTheme {
        UnsupportedDeviceScreen(
            reason = "Missing CPU features: I8MM, BF16",
            modifier = Modifier.fillMaxSize()
        )
    }
}
