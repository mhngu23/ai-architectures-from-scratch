import numpy as np

from utils.layers.linear import Linear, Parameter
from utils.layers.layernorm import LayerNorm
from utils.layers.attention import MultiHeadAttention
from utils.layers.feedforward import PositionwiseFeedForward
from utils.activations.Sigmoid import Sigmoid


class EncoderLayer:
    """Self-attention + feed-forward (FFN), mỗi nhánh đều được bọc trong 1
    residual connection rồi chuẩn hoá kiểu post-norm (`LayerNorm(x +
    Sublayer(x))`) — đúng như bài báo gốc "Attention Is All You Need", khác
    với biến thể pre-norm mà các model đời sau hay dùng.
    """
    def __init__(self, d_model, num_heads, d_ff):
        # d_model: chiều của vector biểu diễn mỗi token (vd 32).
        # num_heads: số "đầu" attention chạy song song, mỗi đầu nhìn vào
        #   d_model/num_heads chiều -> cho phép model học nhiều kiểu quan hệ
        #   giữa các token cùng lúc.
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        # Chuẩn hoá sau khi cộng residual của nhánh self-attention.
        self.norm1 = LayerNorm(d_model)
        # d_ff: chiều ẩn của feed-forward (thường lớn hơn d_model, vd 64-2048),
        # đóng vai trò "xử lý riêng từng token" sau khi attention đã trộn thông tin.
        self.ff = PositionwiseFeedForward(d_model, d_ff)
        # Chuẩn hoá sau khi cộng residual của nhánh feed-forward.
        self.norm2 = LayerNorm(d_model)

    def __call__(self, x, mask=None):
        # Cho phép gọi layer_instance(x) thay vì layer_instance.forward(x),
        # giống cú pháp của PyTorch nn.Module.
        return self.forward(x, mask)

    def forward(self, x, mask=None):
        # x: (batch, seq_len, d_model)
        # Self-attention: mỗi token trong x "nhìn" tất cả token khác trong
        # cùng x để tổng hợp ngữ cảnh (Q=K=V=x).
        attn_out = self.self_attn(x, x, x, mask)
        # Residual connection (x + attn_out) giúp gradient truyền thẳng qua
        # nhiều layer mà không bị tiêu biến, rồi LayerNorm để ổn định phân
        # phối giá trị trước khi vào layer tiếp theo.
        x = self.norm1(x + attn_out)
        # Feed-forward xử lý từng token độc lập (không trộn thông tin giữa
        # các vị trí) qua 2 lớp Linear + activation phi tuyến ở giữa.
        ff_out = self.ff(x)
        # Lại residual + norm cho nhánh feed-forward.
        x = self.norm2(x + ff_out)
        return x

    def backward(self, grad_output):
        # Lan truyền ngược đúng thứ tự ngược lại với forward: norm2 -> ff ->
        # (residual split) -> norm1 -> self_attn -> (residual split).
        # grad_output: dL/d(output của layer này), shape (batch, seq_len, d_model)

        # Bước 1: đi ngược qua norm2. norm2 nhận input là (x1 + ff_out) nên
        # grad_sum2 là gradient chung cho cả x1 (nhánh residual) và ff_out.
        grad_sum2 = self.norm2.backward(grad_output)
        # Đi tiếp ngược qua feed-forward để lấy gradient đối với đầu vào của nó (x1).
        grad_ff_in = self.ff.backward(grad_sum2)
        # Cộng gộp 2 nhánh residual: nhánh đi thẳng (grad_sum2) và nhánh qua ff (grad_ff_in).
        grad_x1 = grad_sum2 + grad_ff_in

        # Bước 2: tương tự cho nhánh self-attention + norm1.
        grad_sum1 = self.norm1.backward(grad_x1)
        # MultiHeadAttention.backward trả về gradient cho Q, K, V đầu vào;
        # ở đây Q=K=V=x (self-attention) nên phải cộng cả 3 lại để ra
        # gradient thật sự của x.
        grad_q, grad_k, grad_v = self.self_attn.backward(grad_sum1)
        # Cộng nhánh residual (grad_sum1, đi thẳng qua phép cộng x + attn_out)
        # với gradient từ nhánh attention.
        grad_x = grad_sum1 + grad_q + grad_k + grad_v
        return grad_x

    def parameters(self):
        # Gom tất cả Parameter (trọng số học được) trong layer này lại thành
        # 1 list phẳng, để optimizer duyệt qua và cập nhật.
        return (self.self_attn.parameters() + self.norm1.parameters()
                + self.ff.parameters() + self.norm2.parameters())


class Encoder:
    """Một chồng gồm `num_layers` EncoderLayer xếp nối tiếp nhau."""
    def __init__(self, d_model, num_heads, d_ff, num_layers):
        # Xếp chồng num_layers EncoderLayer giống hệt nhau về kiến trúc
        # nhưng có trọng số riêng biệt (không share weight).
        self.layers = [EncoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]

    def __call__(self, x, mask=None):
        return self.forward(x, mask)

    def forward(self, x, mask=None):
        # Cho x đi lần lượt qua từng layer, output của layer trước là input
        # của layer sau.
        for layer in self.layers:
            x = layer(x, mask)
        return x

    def backward(self, grad_output):
        # Backward phải đi theo thứ tự NGƯỢC với forward: layer cuối cùng
        # nhận grad_output trước, rồi lan ngược dần về layer đầu tiên.
        for layer in reversed(self.layers):
            grad_output = layer.backward(grad_output)
        return grad_output

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


class DecoderLayer:
    """Masked self-attention + cross-attention (chú ý sang output của
    encoder) + feed-forward (FFN), mỗi nhánh đều được bọc trong 1 residual
    connection rồi post-norm.
    """
    def __init__(self, d_model, num_heads, d_ff):
        # Nhánh 1: self-attention trên chính input của decoder (có thể bị
        # masked để chặn nhìn tương lai trong bài toán sinh chuỗi tự hồi quy).
        self.self_attn = MultiHeadAttention(d_model, num_heads)
        self.norm1 = LayerNorm(d_model)
        # Nhánh 2: cross-attention — Q lấy từ decoder, K/V lấy từ output của
        # encoder. Đây là chỗ decoder "hỏi" encoder để lấy thông tin liên quan.
        self.cross_attn = MultiHeadAttention(d_model, num_heads)
        self.norm2 = LayerNorm(d_model)
        # Nhánh 3: feed-forward giống encoder.
        self.ff = PositionwiseFeedForward(d_model, d_ff)
        self.norm3 = LayerNorm(d_model)

    def __call__(self, x, enc_out, src_mask=None, tgt_mask=None):
        return self.forward(x, enc_out, src_mask, tgt_mask)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        # x: input của decoder, (batch, seq_len_tgt, d_model)
        # enc_out: output của encoder, (batch, seq_len_src, d_model)

        # 1) Masked self-attention: mỗi vị trí trong x nhìn các vị trí khác
        # trong chính x (tgt_mask thường dùng để chặn nhìn "tương lai").
        self_attn_out = self.self_attn(x, x, x, tgt_mask)
        x1 = self.norm1(x + self_attn_out)

        # 2) Cross-attention: Q = x1 (từ decoder), K = V = enc_out (từ
        # encoder) -> decoder "truy vấn" thông tin đã được encoder tổng hợp.
        # src_mask che các vị trí padding phía nguồn (encoder input).
        cross_out = self.cross_attn(x1, enc_out, enc_out, src_mask)
        x2 = self.norm2(x1 + cross_out)

        # 3) Feed-forward như thường lệ.
        ff_out = self.ff(x2)
        x3 = self.norm3(x2 + ff_out)
        return x3

    def backward(self, grad_output):
        """Trả về (grad_x, grad_enc_out): gradient theo input riêng của
        decoder layer này, và gradient theo output của encoder mà nó đã
        cross-attend vào (gradient thứ hai này phải được cộng dồn qua tất
        cả các decoder layer bởi nơi gọi hàm này — xem Decoder.backward).
        """
        # Đi ngược qua norm3 + ff (giống EncoderLayer.backward ở nhánh cuối).
        grad_sum3 = self.norm3.backward(grad_output)
        grad_ff_in = self.ff.backward(grad_sum3)
        grad_x2 = grad_sum3 + grad_ff_in

        # Đi ngược qua norm2 + cross-attention.
        grad_sum2 = self.norm2.backward(grad_x2)
        # cross_attn nhận Q=x1, K=V=enc_out nên backward trả về 3 gradient
        # riêng biệt: grad_q1 (cho x1), grad_k_enc và grad_v_enc (cho enc_out).
        grad_q1, grad_k_enc, grad_v_enc = self.cross_attn.backward(grad_sum2)
        # Nhánh residual của x1: cộng đường đi thẳng (grad_sum2) với gradient
        # qua nhánh attention (grad_q1, vì x1 chỉ đóng vai trò Q ở đây).
        grad_x1 = grad_sum2 + grad_q1
        # enc_out đóng vai trò cả K lẫn V trong cross-attention nên gradient
        # của nó là tổng 2 đường (không có residual vì enc_out không được
        # cộng trực tiếp vào output của layer này).
        grad_enc_out = grad_k_enc + grad_v_enc

        # Đi ngược qua norm1 + masked self-attention, y hệt EncoderLayer.
        grad_sum1 = self.norm1.backward(grad_x1)
        grad_q, grad_k, grad_v = self.self_attn.backward(grad_sum1)
        grad_x = grad_sum1 + grad_q + grad_k + grad_v

        # Trả về gradient cho input riêng của layer này (grad_x) VÀ gradient
        # đóng góp vào enc_out (grad_enc_out) — cái sau sẽ được Decoder cộng
        # dồn từ mọi layer vì tất cả layer đều đọc chung 1 enc_out.
        return grad_x, grad_enc_out

    def parameters(self):
        return (self.self_attn.parameters() + self.norm1.parameters()
                + self.cross_attn.parameters() + self.norm2.parameters()
                + self.ff.parameters() + self.norm3.parameters())


class Decoder:
    """Một chồng gồm `num_layers` DecoderLayer xếp nối tiếp nhau."""
    def __init__(self, d_model, num_heads, d_ff, num_layers):
        self.layers = [DecoderLayer(d_model, num_heads, d_ff) for _ in range(num_layers)]

    def __call__(self, x, enc_out, src_mask=None, tgt_mask=None):
        return self.forward(x, enc_out, src_mask, tgt_mask)

    def forward(self, x, enc_out, src_mask=None, tgt_mask=None):
        # enc_out giữ nguyên (không đổi) qua mọi layer — mỗi DecoderLayer
        # đều cross-attend vào CÙNG một enc_out; chỉ x (trạng thái decoder)
        # được cập nhật dần qua từng layer.
        for layer in self.layers:
            x = layer(x, enc_out, src_mask, tgt_mask)
        return x

    def backward(self, grad_output):
        """Trả về (grad_x, grad_enc_out) — grad_enc_out là TỔNG gradient
        cross-attention đóng góp bởi từng layer, vì tất cả các layer đều
        đọc chung một encoder output.
        """
        # Vì mọi layer đều dùng chung enc_out ở forward, nên trong backward
        # gradient của enc_out phải được CỘNG DỒN (tích luỹ) từ tất cả các
        # layer, không phải chỉ lấy của layer cuối.
        grad_enc_out_total = 0.0
        for layer in reversed(self.layers):
            # Mỗi layer.backward trả về gradient cho input riêng của nó
            # (dùng làm grad_output cho layer trước đó trong vòng lặp) và
            # phần đóng góp vào gradient của enc_out.
            grad_output, grad_enc_out = layer.backward(grad_output)
            grad_enc_out_total = grad_enc_out_total + grad_enc_out
        return grad_output, grad_enc_out_total

    def parameters(self):
        params = []
        for layer in self.layers:
            params.extend(layer.parameters())
        return params


class FeatureTokenizer:
    """Biến một bảng số dạng (batch, num_features) thành một chuỗi "token"
    dạng (batch, num_features, d_model), mỗi cột dữ liệu ứng với 1 token,
    để cơ chế attention của Transformer có thể áp dụng lên dữ liệu dạng bảng.

    Mỗi feature (cột) có riêng 1 phép biến đổi tuyến tính (affine) học được
    scalar -> d_model (một `Linear(1, d_model)` riêng cho từng cột), vì ví
    dụ "Glucose" và "BMI" nằm trên thang đo khác nhau nên không nên dùng
    chung 1 phép chiếu. Ý tưởng này theo "feature-tokenizer" của
    FT-Transformer (Gorishniy et al., 2021).
    """
    def __init__(self, num_features, d_model):
        self.num_features = num_features
        self.d_model = d_model
        # Mỗi cột dữ liệu (feature) có 1 Linear(1, d_model) RIÊNG — nghĩa là
        # scalar của cột "Glucose" được nhân với 1 vector trọng số khác với
        # scalar của cột "BMI", vì 2 cột đó có ý nghĩa/thang đo khác nhau.
        self.embeddings = [Linear(1, d_model) for _ in range(num_features)]

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # x: (batch, num_features) -> tokens: (batch, num_features, d_model)
        # Với mỗi cột j, lấy x[:, j:j+1] (shape (batch, 1)) rồi chiếu lên
        # d_model chiều bằng Linear riêng của cột đó -> ra (batch, d_model).
        tokens = [self.embeddings[j](x[:, j:j + 1]) for j in range(self.num_features)]
        # Xếp num_features vector đó lại theo trục mới (axis=1) để tạo thành
        # 1 "chuỗi" gồm num_features token, giống 1 câu gồm nhiều từ.
        return np.stack(tokens, axis=1)

    def backward(self, grad_output):
        # grad_output: (batch, num_features, d_model) -> (batch, num_features)
        # Với mỗi cột j, lấy gradient tương ứng grad_output[:, j, :] và lan
        # ngược qua đúng Linear riêng của cột đó.
        grad_cols = [self.embeddings[j].backward(grad_output[:, j, :]) for j in range(self.num_features)]
        # Ghép các gradient cột (mỗi cột shape (batch, 1)) lại theo trục cột
        # để khôi phục shape gốc (batch, num_features).
        return np.concatenate(grad_cols, axis=1)

    def parameters(self):
        params = []
        for embedding in self.embeddings:
            params.append(embedding.W)
            params.append(embedding.b)
        return params


class TabularTransformer:
    """Transformer dạng encoder-decoder cho bài toán phân loại nhị phân trên
    dữ liệu bảng (tabular).

    Với dữ liệu bảng thì không có khái niệm "chuỗi mục tiêu" (target
    sequence) tự nhiên như dịch máy, nên phần decoder được điều chỉnh lại
    thay vì dùng để sinh chuỗi tự hồi quy như bình thường:
    - Encoder: FeatureTokenizer biến mỗi cột trong `num_features` cột thành
      1 token d_model chiều; self-attention giúp encoder học được tương tác
      giữa các feature (vd Glucose x BMI).
    - Decoder: chỉ dùng 1 token "truy vấn" (query) duy nhất, học được (giống
      kiểu [CLS] trong BERT hoặc "object query" trong DETR). Token này đi
      qua masked self-attention (gần như vô nghĩa vì chỉ có 1 token) rồi
      cross-attend sang các feature-token của encoder để gom (pool) tất cả
      lại thành 1 vector tóm tắt duy nhất.
    - Head: Linear(d_model, 1) + Sigmoid áp lên token output của decoder để
      ra xác suất thuộc lớp dương, huấn luyện bằng BCE loss.

    Cách làm này vẫn giữ đầy đủ các thành phần (encoder, decoder,
    cross-attention) đúng như một "complete encoder-decoder model" trong lộ
    trình học, đồng thời áp dụng được trực tiếp lên bộ dữ liệu Pima diabetes.
    """
    def __init__(self, num_features, d_model=32, num_heads=4, d_ff=64, num_layers=2):
        # Biến mỗi hàng (num_features cột) thành num_features "token".
        self.tokenizer = FeatureTokenizer(num_features, d_model)
        # Encoder học quan hệ/tương tác GIỮA các feature với nhau (attention
        # cho phép "Glucose" chú ý tới "BMI" chẳng hạn).
        self.encoder = Encoder(d_model, num_heads, d_ff, num_layers)
        # Decoder ở đây không sinh chuỗi — nó chỉ dùng 1 token truy vấn duy
        # nhất để "gom" (pool) toàn bộ thông tin từ encoder thành 1 vector.
        self.decoder = Decoder(d_model, num_heads, d_ff, num_layers)
        # query: 1 vector học được, đóng vai trò giống token [CLS] trong
        # BERT hoặc "object query" trong DETR — không phụ thuộc dữ liệu đầu
        # vào, chỉ học cách "hỏi" encoder câu hỏi hữu ích nhất cho việc
        # phân loại. Khởi tạo nhỏ (*0.02) để không lấn át tín hiệu ban đầu.
        self.query = Parameter(np.random.randn(1, 1, d_model) * 0.02)
        # Lớp tuyến tính cuối, chiếu từ d_model xuống 1 số thực (logit).
        self.head = Linear(d_model, 1)
        # Sigmoid biến logit thành xác suất thuộc lớp dương, dùng cho BCE loss.
        self.sigmoid = Sigmoid()

    def __call__(self, x):
        return self.forward(x)

    def forward(self, x):
        # x: (batch, num_features) — 1 hàng dữ liệu bảng, mỗi cột 1 giá trị số.
        batch = x.shape[0]
        # Biến bảng thành chuỗi token: (batch, num_features, d_model).
        tokens = self.tokenizer(x)
        # Encoder trộn thông tin giữa các feature-token với nhau qua self-attention.
        enc_out = self.encoder(tokens)

        # self.query có shape (1, 1, d_model) — dùng chung cho mọi mẫu, nên
        # phải nhân bản (repeat) theo batch để có shape (batch, 1, d_model)
        # khớp với enc_out.
        query = np.repeat(self.query.data, batch, axis=0)  # (batch, 1, d_model)
        # Decoder: self-attention trên 1 token (gần như no-op vì chỉ có 1 vị
        # trí để "nhìn"), sau đó cross-attention để token query hút thông
        # tin từ toàn bộ enc_out (mọi feature-token).
        dec_out = self.decoder(query, enc_out)

        # dec_out có shape (batch, 1, d_model); bỏ trục "seq_len=1" để lấy
        # vector tổng hợp (batch, d_model) — đây là bản tóm tắt toàn bộ hàng
        # dữ liệu sau khi đã đi qua attention.
        pooled = dec_out[:, 0, :]  # (batch, d_model)
        # Chiếu tuyến tính xuống 1 giá trị logit cho mỗi mẫu.
        logits = self.head(pooled)
        # Sigmoid -> xác suất trong khoảng (0, 1), dùng để phân loại nhị phân.
        return self.sigmoid(logits)

    def backward(self, grad_output):
        # grad_output: dL/d(output sigmoid), thường đến từ đạo hàm của BCE loss.

        # Lan ngược qua sigmoid rồi qua lớp head (Linear(d_model, 1)).
        grad_logits = self.sigmoid.backward(grad_output)
        grad_pooled = self.head.backward(grad_logits)
        # pooled được lấy ra bằng dec_out[:, 0, :] (bỏ trục seq_len=1), nên
        # để lan ngược đúng shape ta phải "thêm lại" trục đó bằng newaxis.
        grad_dec_out = grad_pooled[:, np.newaxis, :]  # (batch, 1, d_model)

        # Decoder.backward trả về gradient cho input của nó (grad_query —
        # gradient theo từng mẫu trong batch của token query đã nhân bản)
        # và gradient cộng dồn cho enc_out.
        grad_query, grad_enc_out = self.decoder.backward(grad_dec_out)
        # self.query là 1 tham số DÙNG CHUNG cho mọi mẫu trong batch (đã bị
        # np.repeat ở forward), nên theo quy tắc lan truyền ngược khi 1 giá
        # trị được "phát" (broadcast) ra nhiều bản sao, gradient của nó phải
        # là TỔNG gradient của tất cả các bản sao đó -> sum theo axis=0.
        grad_query_sum = grad_query.sum(axis=0, keepdims=True)
        # Cộng dồn vào self.query.grad (giống các Parameter khác) thay vì
        # ghi đè, để hỗ trợ trường hợp backward được gọi nhiều lần trước khi
        # optimizer step (gradient accumulation).
        self.query.grad = grad_query_sum if self.query.grad is None else self.query.grad + grad_query_sum

        # Lan tiếp gradient của enc_out ngược qua Encoder, rồi ngược qua
        # FeatureTokenizer để ra gradient theo đúng shape đầu vào ban đầu (batch, num_features).
        grad_tokens = self.encoder.backward(grad_enc_out)
        grad_x = self.tokenizer.backward(grad_tokens)
        return grad_x

    def parameters(self):
        # Gom TẤT CẢ tham số học được của toàn bộ mô hình (tokenizer,
        # encoder, decoder, query, head) thành 1 list phẳng để optimizer
        # (vd SGD/Adam viết tay) duyệt qua và cập nhật từng Parameter.
        params = self.tokenizer.parameters() + self.encoder.parameters() + self.decoder.parameters()
        params.append(self.query)
        params.append(self.head.W)
        params.append(self.head.b)
        return params
