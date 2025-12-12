import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        model_name = "sentence-transformers/all-MiniLM-L6-v2"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def execute(self, requests):
        responses = []

        for request in requests:
            # Get tensor
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            raw = input_tensor.as_numpy()  # shape [batch, 1]

            # Flatten and safely decode
            texts = []
            for item in raw.reshape(-1):
                if isinstance(item, (bytes, bytearray)):
                    texts.append(item.decode("utf-8"))
                else:
                    texts.append(str(item))

            # Tokenize
            tokens = self.tokenizer(
                texts,
                padding=True,
                truncation=True,
                return_tensors="pt"
            )

            with torch.no_grad():
                out = self.model(**tokens)
                embeddings = out.last_hidden_state.mean(dim=1)

            # Convert to FP32 numpy
            embeddings = embeddings.cpu().numpy().astype(np.float32)

            # Build response
            output_tensor = pb_utils.Tensor("OUTPUT", embeddings)
            responses.append(pb_utils.InferenceResponse(
                output_tensors=[output_tensor]
            ))

        return responses

