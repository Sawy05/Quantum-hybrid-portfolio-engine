"""Hybrid Quantum-Classical Neural Network integrating PyTorch and PennyLane VQC."""

from __future__ import annotations

from typing import Dict, List, Tuple

import numpy as np
import pennylane as qml
import torch
import torch.nn as nn
from torch.utils.data import DataLoader, TensorDataset

from src.qml_models.circuits import QuantumCircuitFactory


class HybridQuantumClassifier(nn.Module):
    """Hybrid Classical-Quantum Model:

    Input (Batch, Seq_Len, Dim)
      -> Classical Temporal Extractor (LSTM or Conv1D)
      -> Linear Projection to [-pi, pi] (Batch, N_Qubits)
      -> PennyLane VQC TorchLayer (Pauli-Z Expectation Values in [-1, 1])
      -> Dense Classification Head (Batch, Num_Classes)
    """

    def __init__(
        self,
        n_qubits: int = 4,
        circuit_depth: int = 3,
        feature_dim: int = 4,
        hidden_dim: int = 16,
        num_classes: int = 3,
        encoding_type: str = "angle",
        entanglement_topology: str = "ring",
        diff_method: str = "backprop",
        device_name: str = "default.qubit",
    ):
        super().__init__()
        self.n_qubits = n_qubits
        self.circuit_depth = circuit_depth
        self.feature_dim = feature_dim
        self.hidden_dim = hidden_dim
        self.num_classes = num_classes

        # 1. Classical Temporal Feature Extractor (LSTM)
        self.lstm = nn.LSTM(
            input_size=feature_dim,
            hidden_size=hidden_dim,
            num_layers=1,
            batch_first=True,
        )

        # 2. Linear projection from LSTM latent representation to Quantum Angle Space [-pi, pi]
        self.classical_to_quantum = nn.Sequential(
            nn.Linear(hidden_dim, n_qubits),
            nn.Hardtanh(min_val=-float(np.pi), max_val=float(np.pi)),
        )

        # 3. Quantum Variational Layer
        self.circuit_factory = QuantumCircuitFactory(
            n_qubits=n_qubits,
            circuit_depth=circuit_depth,
            encoding_type=encoding_type,  # type: ignore
            entanglement_topology=entanglement_topology,  # type: ignore
            diff_method=diff_method,
            device_name=device_name,
        )

        qnode = self.circuit_factory.create_vqc_qnode(interface="torch")
        weight_shapes = {"weights": self.circuit_factory.weight_shape()}
        self.quantum_layer = qml.qnn.TorchLayer(qnode, weight_shapes)

        # 4. Dense Classical Classification Head
        # Concatenate classical hidden features + quantum expectation values for residual stability
        self.classifier = nn.Sequential(
            nn.Linear(n_qubits + hidden_dim, hidden_dim),
            nn.ReLU(),
            nn.Dropout(p=0.1),
            nn.Linear(hidden_dim, num_classes),
        )

    def forward(self, x: torch.Tensor) -> Tuple[torch.Tensor, torch.Tensor, torch.Tensor]:
        """Forward pass.

        Returns:
            logits: (Batch, Num_Classes)
            q_angles: (Batch, N_Qubits) in [-pi, pi]
            q_expvals: (Batch, N_Qubits) in [-1, 1]
        """
        # x: (Batch, Seq_Len, Feature_Dim)
        lstm_out, (hn, _) = self.lstm(x)
        # Use last hidden state: (Batch, Hidden_Dim)
        last_hidden = hn[-1]

        # Classical-to-Quantum Projection
        q_angles = self.classical_to_quantum(last_hidden)

        # Quantum Layer expectation values: (Batch, N_Qubits)
        q_expvals = self.quantum_layer(q_angles)

        # Combined features for classification head
        combined = torch.cat([q_expvals, last_hidden], dim=-1)
        logits = self.classifier(combined)

        return logits, q_angles, q_expvals

    def fit_model(
        self,
        X_train: np.ndarray,
        y_train: np.ndarray,
        epochs: int = 10,
        batch_size: int = 16,
        lr: float = 0.01,
        verbose: bool = True,
    ) -> Dict[str, List[float]]:
        """Trains the hybrid model end-to-end using Adam and cross-entropy loss."""
        dataset = TensorDataset(
            torch.tensor(X_train, dtype=torch.float32),
            torch.tensor(y_train, dtype=torch.long),
        )
        loader = DataLoader(dataset, batch_size=batch_size, shuffle=True)

        criterion = nn.CrossEntropyLoss()
        optimizer = torch.optim.Adam(self.parameters(), lr=lr, weight_decay=1e-4)

        history: Dict[str, List[float]] = {"loss": [], "accuracy": []}

        self.train()
        for epoch in range(epochs):
            total_loss = 0.0
            correct = 0
            total = 0

            for batch_x, batch_y in loader:
                optimizer.zero_grad()
                logits, _, _ = self(batch_x)
                loss = criterion(logits, batch_y)
                loss.backward()
                optimizer.step()

                total_loss += loss.item() * len(batch_y)
                preds = torch.argmax(logits, dim=-1)
                correct += (preds == batch_y).sum().item()
                total += len(batch_y)

            epoch_loss = total_loss / max(1, total)
            epoch_acc = correct / max(1, total)
            history["loss"].append(epoch_loss)
            history["accuracy"].append(epoch_acc)

            if verbose and ((epoch + 1) % max(1, epochs // 5) == 0 or epoch == epochs - 1):
                print(
                    f"Epoch [{epoch + 1}/{epochs}] - Loss: {epoch_loss:.4f} - Acc: {epoch_acc * 100:.2f}%"
                )

        return history

    @torch.no_grad()
    def predict_regime(
        self, x: np.ndarray
    ) -> Tuple[np.ndarray, np.ndarray, np.ndarray, np.ndarray]:
        """Predicts regime probabilities, predicted class, rotation angles, and Pauli-Z expectation values."""
        self.eval()
        if x.ndim == 2:
            x = np.expand_dims(x, axis=0)
        tensor_x = torch.tensor(x, dtype=torch.float32)
        logits, q_angles, q_expvals = self(tensor_x)
        probs = torch.softmax(logits, dim=-1).cpu().numpy()
        preds = np.argmax(probs, axis=-1)
        return preds, probs, q_angles.cpu().numpy(), q_expvals.cpu().numpy()
