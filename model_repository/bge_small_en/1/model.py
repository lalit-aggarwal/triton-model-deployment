import numpy as np
import torch
from transformers import AutoTokenizer, AutoModel
import triton_python_backend_utils as pb_utils

class TritonPythonModel:
    def initialize(self, args):
        model_name = "BAAI/bge-small-en"
        self.tokenizer = AutoTokenizer.from_pretrained(model_name)
        self.model = AutoModel.from_pretrained(model_name)

    def execute(self, requests):
        responses = []

        for request in requests:
            input_tensor = pb_utils.get_input_tensor_by_name(request, "INPUT")
            raw = input_tensor.as_numpy().reshape(-1)

            texts = [
                x.decode("utf-8") if isinstance(x, (bytes, bytearray)) else str(x)
                for x in raw
            ]

            tokens = self.tokenizer(texts, padding=True, truncation=True, return_tensors="pt")

            with torch.no_grad():
                out = self.model(**tokens)
                embeddings = out.last_hidden_state.mean(dim=1)

            embeddings = embeddings.cpu().numpy().astype(np.float32)

            responses.append(
                pb_utils.InferenceResponse(
                    output_tensors=[pb_utils.Tensor("OUTPUT", embeddings)]
                )
            )

        return responses
