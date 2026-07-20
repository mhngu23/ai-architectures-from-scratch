"""Kiểm tra gradient bằng phương pháp số (numerical/finite-difference) cho
các khối xây dựng nên encoder/decoder.

Mọi module ở đây đều tự viết tay `backward()` (không dùng autograd của
PyTorch/TensorFlow), nên cách duy nhất đáng tin cậy để kiểm chứng công thức
đạo hàm viết tay có đúng hay không là so sánh nó với gradient tính bằng
phương pháp sai phân trung tâm (finite difference) trên một hàm loss vô
hướng (scalar). Shape được cố tình giữ rất nhỏ (batch=2, seq=3, d_model=4)
chỉ để vòng lặp sai phân hữu hạn (vốn có độ phức tạp O(n), phải chạy forward
2 lần cho MỖI phần tử của x) chạy đủ nhanh.
"""
import numpy as np

from utils.layers.layernorm import LayerNorm
from utils.layers.attention import ScaledDotProductAttention, MultiHeadAttention
from utils.layers.feedforward import PositionwiseFeedForward
from models.transformer.model import EncoderLayer, DecoderLayer, FeatureTokenizer, TabularTransformer

# Seed cố định để mọi lần chạy test đều dùng cùng 1 bộ trọng số/input ngẫu
# nhiên -> kết quả test có thể tái lập (reproducible), không "may rủi" đậu/rớt.
np.random.seed(0)
# eps: bước nhiễu (perturbation) rất nhỏ dùng để xấp xỉ đạo hàm bằng sai phân.
EPS = 1e-4
# TOL: sai số tương đối tối đa cho phép giữa gradient viết tay (analytic) và
# gradient tính số (numeric) để coi là "khớp" — không thể yêu cầu bằng tuyệt
# đối vì bản thân phép sai phân hữu hạn cũng chỉ là một xấp xỉ (có sai số
# bậc O(eps^2)), cộng thêm sai số làm tròn dấu phẩy động.
TOL = 1e-3


def numerical_grad(f, x, eps=EPS):
    """Tính gradient bằng sai phân trung tâm (central difference) của hàm
    trả về 1 số vô hướng `f`, theo TỪNG phần tử của mảng `x`.

    Công thức: df/dx_i ≈ (f(x_i + eps) - f(x_i - eps)) / (2 * eps)
    Đây là công thức sai phân "trung tâm" (dùng cả 2 phía +eps và -eps),
    chính xác hơn sai phân "một phía" (forward difference) vì sai số của nó
    là bậc O(eps^2) thay vì O(eps).
    """
    # grad: mảng kết quả, cùng shape với x, mỗi phần tử là đạo hàm riêng
    # (partial derivative) của f theo đúng phần tử tương ứng trong x.
    grad = np.zeros_like(x)
    # np.nditer với multi_index cho phép duyệt qua TỪNG phần tử của mảng x
    # dù x có bao nhiêu chiều đi nữa (1D, 2D, 3D, 4D...), và biết được toạ
    # độ (multi_index) của phần tử đang xét để có thể sửa/khôi phục giá trị.
    it = np.nditer(x, flags=['multi_index'])
    for _ in it:
        idx = it.multi_index
        # Lưu lại giá trị gốc để khôi phục sau khi tính xong đạo hàm tại vị
        # trí này — không được để lại "vết" nhiễu trong x cho lần lặp sau.
        orig = x[idx]

        # Nhích x[idx] lên một chút (+eps), chạy lại toàn bộ forward+loss để
        # xem loss thay đổi thế nào -> đây là f(x + eps) trong công thức.
        x[idx] = orig + eps
        plus = f()

        # Nhích x[idx] xuống một chút (-eps) -> f(x - eps).
        x[idx] = orig - eps
        minus = f()

        # Khôi phục lại giá trị gốc, KHÔNG được quên bước này vì các phép
        # gọi f() tiếp theo (cho phần tử khác) cần x ở trạng thái nguyên bản.
        x[idx] = orig

        # Áp dụng công thức sai phân trung tâm để ước lượng đạo hàm riêng
        # tại đúng vị trí idx.
        grad[idx] = (plus - minus) / (2 * eps)
    return grad


def relative_error(a, b):
    # Sai số TƯƠNG ĐỐI giữa 2 mảng gradient a và b (thay vì sai số tuyệt
    # đối |a-b|), vì gradient có thể có độ lớn (magnitude) rất khác nhau
    # giữa các bài test (vd gradient của LayerNorm rất nhỏ, của FFN có thể
    # lớn hơn nhiều) — dùng sai số tương đối giúp ngưỡng TOL áp dụng công
    # bằng cho mọi trường hợp. Cộng thêm 1e-8 ở mẫu số để tránh chia cho 0
    # khi cả 2 gradient đều gần bằng 0 tại vị trí đó.
    # Lấy max trên toàn mảng: chỉ cần 1 phần tử sai lệch nhiều là đã coi là fail.
    return np.max(np.abs(a - b) / (np.abs(a) + np.abs(b) + 1e-8))


def check_grad(name, analytic, numeric):
    # analytic: gradient do chính module tự tính (bằng backward() viết tay).
    # numeric: gradient ước lượng bằng sai phân hữu hạn (coi như "đáp án
    # tham khảo" độc lập, không phụ thuộc vào công thức backward() có đúng
    # hay không).
    err = relative_error(analytic, numeric)
    status = "OK" if err < TOL else "FAIL"
    # In ra ngay cả khi PASS, để khi chạy file trực tiếp (__main__) có thể
    # nhìn thấy toàn bộ danh sách các gradient đã được kiểm tra, không chỉ
    # khi có lỗi.
    print(f"  [{status}] {name}: max relative error = {err:.2e}")
    # assert: nếu dùng pytest thì đây chính là điều kiện làm test FAIL; nếu
    # chạy trực tiếp bằng __main__ thì AssertionError sẽ dừng chương trình
    # ngay tại module đang bị sai, kèm thông báo rõ ràng module nào lỗi.
    assert err < TOL, f"{name} gradient check failed (err={err:.2e})"


def loss_and_grad(out):
    """L = 0.5 * sum(out^2)  =>  dL/dout = out."""
    # Hàm loss "giả" (không phải BCE/MSE thật) dùng riêng cho việc kiểm tra
    # gradient: vì dL/dout = out (đạo hàm cực kỳ đơn giản, tính tay được
    # ngay), ta có thể dùng out CHÍNH NÓ làm grad_output ban đầu để bắt đầu
    # lan truyền ngược qua module đang test, mà không cần phụ thuộc vào
    # BCELoss/MSELoss (những phần cũng cần được test riêng).
    return 0.5 * np.sum(out ** 2), out


def check_module(module, inputs, input_names, param_names=None):
    """Chạy forward, backward, rồi kiểm tra bằng phương pháp số gradient
    của TỪNG mảng trong `inputs`, cộng với TỪNG Parameter mà
    `module.parameters()` trả về (nếu có truyền `param_names`, độ dài/thứ
    tự phải khớp với `parameters()`).
    """
    def forward_all():
        # Hàm "wrapper" chạy lại đúng 1 lượt forward + tính loss vô hướng,
        # dùng làm đối số `f` cho numerical_grad ở trên. Được định nghĩa lại
        # mỗi lần gọi check_module để "đóng" (closure) đúng biến `module` và
        # `inputs` hiện tại của lần gọi này.
        out = module.forward(*inputs)
        loss, _ = loss_and_grad(out)
        return loss

    # ----- Nhánh gradient viết tay (analytic) -----
    out = module.forward(*inputs)
    _, grad_out = loss_and_grad(out)
    # backward() có thể trả về 1 mảng DUY NHẤT (module chỉ có 1 input, vd
    # LayerNorm) hoặc 1 TUPLE nhiều mảng (module có nhiều input, vd Q/K/V
    # của attention) — chuẩn hoá cả 2 trường hợp về dạng tuple để xử lý
    # đồng nhất bằng vòng lặp zip() bên dưới.
    result = module.backward(grad_out)
    analytic_grads = result if isinstance(result, tuple) else (result,)

    # ----- So sánh gradient của từng INPUT -----
    for name, x, analytic in zip(input_names, inputs, analytic_grads):
        # Với mỗi input x, tính gradient bằng sai phân hữu hạn (độc lập
        # hoàn toàn với backward()) rồi so sánh với gradient viết tay.
        numeric = numerical_grad(forward_all, x)
        check_grad(name, analytic, numeric)

    # ----- So sánh gradient của từng PARAMETER (W, b, gamma, beta...) -----
    if param_names is not None:
        for name, p in zip(param_names, module.parameters()):
            def forward_all_reset():
                # Về bản chất giống hệt forward_all(), định nghĩa lại ở đây
                # chỉ để có 1 closure riêng cho từng Parameter trong vòng
                # lặp (tránh nhầm lẫn biến giữa các lần lặp, dù ở đây không
                # bắt buộc vì forward_all không giữ trạng thái theo p).
                return forward_all()
            # p.data là giá trị hiện tại của tham số; numerical_grad sẽ tự
            # nhiễu +eps/-eps ngay trên p.data (đối tượng Parameter được
            # truyền theo tham chiếu) rồi khôi phục lại sau khi tính xong.
            numeric = numerical_grad(forward_all_reset, p.data)
            # p.grad là gradient mà module.backward() ở trên đã TÍCH LUỸ
            # (side effect) vào Parameter này khi được gọi.
            check_grad(name, p.grad, numeric)


def test_layernorm():
    print("LayerNorm")
    np.random.seed(0)
    ln = LayerNorm(4)
    # x: (batch=2, seq=3, d_model=4) — LayerNorm chuẩn hoá trên trục cuối
    # (d_model), nên input có thêm trục "seq" để test luôn cả trường hợp
    # nhiều hơn 2 chiều (không chỉ (batch, dim) thông thường).
    x = np.random.randn(2, 3, 4)
    # LayerNorm có 2 Parameter: gamma và beta (xem parameters() trong
    # layernorm.py) -> đặt tên chung chung "param[0]", "param[1]" vì test
    # này chỉ cần biết THỨ TỰ khớp với module.parameters(), không cần tên
    # gợi nhớ cụ thể.
    check_module(ln, [x], ["x"], [f"param[{i}]" for i in range(2)])


def test_scaled_dot_product_attention():
    print("ScaledDotProductAttention")
    np.random.seed(0)
    attn = ScaledDotProductAttention()
    # Q, K, V: (batch=2, heads=2, seq=3, d_k=4) — shape đúng như
    # ScaledDotProductAttention kỳ vọng SAU KHI MultiHeadAttention đã tách
    # d_model thành các head riêng biệt (xem attention.py). Module này
    # không có Parameter học được (chỉ là phép tính thuần tuý Q,K,V ->
    # attention), nên không truyền param_names.
    Q = np.random.randn(2, 2, 3, 4)
    K = np.random.randn(2, 2, 3, 4)
    V = np.random.randn(2, 2, 3, 4)
    check_module(attn, [Q, K, V], ["Q", "K", "V"])


def test_multihead_attention():
    print("MultiHeadAttention")
    np.random.seed(0)
    mha = MultiHeadAttention(d_model=4, num_heads=2)
    # Q_in/K_in/V_in: (batch=2, seq=3, d_model=4) — đây là input TRƯỚC KHI
    # bị tách head (việc tách head diễn ra bên trong MultiHeadAttention).
    Q = np.random.randn(2, 3, 4)
    K = np.random.randn(2, 3, 4)
    V = np.random.randn(2, 3, 4)
    # MultiHeadAttention có 4 phép chiếu Linear (W_q, W_k, W_v, W_o), mỗi
    # phép chiếu góp 2 Parameter (W và b) -> 4 * 2 = 8 param.
    check_module(mha, [Q, K, V], ["Q_in", "K_in", "V_in"], [f"param[{i}]" for i in range(8)])


def test_feedforward():
    print("PositionwiseFeedForward")
    np.random.seed(0)
    ff = PositionwiseFeedForward(d_model=4, d_ff=6)
    x = np.random.randn(2, 3, 4)
    # PositionwiseFeedForward = Linear(d_model,d_ff) -> ReLU -> Linear(d_ff,
    # d_model), tức 2 lớp Linear * 2 param (W,b) mỗi lớp = 4 param.
    check_module(ff, [x], ["x"], [f"param[{i}]" for i in range(4)])


def test_encoder_layer():
    print("EncoderLayer")
    np.random.seed(0)
    layer = EncoderLayer(d_model=4, num_heads=2, d_ff=6)
    x = np.random.randn(2, 3, 4)
    # Khác với các test trên (đếm tay số param), ở đây lấy trực tiếp
    # len(layer.parameters()) — vì EncoderLayer gộp nhiều sub-module
    # (self_attn + norm1 + ff + norm2), đếm tay dễ sai nên tin tưởng luôn
    # vào chính hàm parameters() của module (bản thân test này KHÔNG kiểm
    # tra parameters() có đúng/đủ hay không, chỉ kiểm tra gradient của
    # những gì nó trả về).
    n_params = len(layer.parameters())
    check_module(layer, [x], ["x"], [f"param[{i}]" for i in range(n_params)])


def test_decoder_layer():
    print("DecoderLayer")
    np.random.seed(0)
    layer = DecoderLayer(d_model=4, num_heads=2, d_ff=6)
    # x: input riêng của decoder — cố tình cho seq_len=1 (khác với seq_len=3
    # của encoder output) để test luôn trường hợp seq_len KHÔNG BẰNG NHAU
    # giữa Q (từ decoder) và K/V (từ encoder) trong cross-attention, đúng
    # như cách TabularTransformer dùng DecoderLayer trong thực tế (query
    # token seq_len=1, enc_out seq_len=num_features).
    x = np.random.randn(2, 1, 4)
    # enc_out: output giả lập của encoder, seq_len=3.
    enc_out = np.random.randn(2, 3, 4)

    # DecoderLayer.backward() trả về CẶP (grad_x, grad_enc_out) — khác cấu
    # trúc input/output của check_module() (vốn giả định input là 1 list
    # duy nhất đưa hết vào forward), nên test này viết tay logic riêng thay
    # vì gọi check_module().
    def forward_all():
        out = layer.forward(x, enc_out)
        loss, _ = loss_and_grad(out)
        return loss

    out = layer.forward(x, enc_out)
    _, grad_out = loss_and_grad(out)
    grad_x, grad_enc = layer.backward(grad_out)

    # Kiểm tra riêng biệt gradient theo x (input decoder) và theo enc_out
    # (input encoder) — đây chính là phép test quan trọng nhất để xác nhận
    # DecoderLayer.backward() tách đúng phần "residual" (chỉ về x1) và phần
    # "cross-attention K/V" (chỉ về enc_out), như đã giải thích trong
    # models/transformer/model.py.
    check_grad("x", grad_x, numerical_grad(forward_all, x))
    check_grad("enc_out", grad_enc, numerical_grad(forward_all, enc_out))
    for i, p in enumerate(layer.parameters()):
        check_grad(f"param[{i}]", p.grad, numerical_grad(forward_all, p.data))


def test_feature_tokenizer():
    print("FeatureTokenizer")
    np.random.seed(0)
    tok = FeatureTokenizer(num_features=3, d_model=4)
    # x: (batch=2, num_features=3) — bảng số giả lập, KHÔNG phải chuỗi
    # token, vì FeatureTokenizer nhận input dạng bảng (batch, num_features)
    # và tự biến nó thành chuỗi token (batch, num_features, d_model).
    x = np.random.randn(2, 3)
    # 3 feature, mỗi feature có 1 Linear(1, d_model) riêng -> 3 * 2 (W,b) = 6 param.
    check_module(tok, [x], ["x"], [f"param[{i}]" for i in range(6)])


def test_tabular_transformer():
    print("TabularTransformer (full encoder-decoder model)")
    np.random.seed(0)
    # num_layers=1 (thay vì 2 như trong notebook demo) để giữ mô hình nhỏ
    # nhất có thể — mục tiêu của test này là kiểm tra TOÀN BỘ pipeline
    # (tokenizer -> encoder -> decoder -> head -> sigmoid) lan truyền ngược
    # đúng đầu-đến-cuối, không phải để đánh giá chất lượng model.
    model = TabularTransformer(num_features=3, d_model=4, num_heads=2, d_ff=6, num_layers=1)
    x = np.random.randn(2, 3)
    n_params = len(model.parameters())
    check_module(model, [x], ["x"], [f"param[{i}]" for i in range(n_params)])


if __name__ == "__main__":
    # Chạy tuần tự từng test khi gọi trực tiếp `python test_transformer_modules.py`
    # (không cần pytest) — hữu ích để xem TOÀN BỘ log "[OK]/[FAIL]" của mọi
    # gradient được kiểm tra, thay vì chỉ thấy PASS/FAIL gộp như pytest.
    for fn in [
        test_layernorm,
        test_scaled_dot_product_attention,
        test_multihead_attention,
        test_feedforward,
        test_encoder_layer,
        test_decoder_layer,
        test_feature_tokenizer,
        test_tabular_transformer,
        # test_autoencoder,  # Uncomment if you want to include Autoencoder tests
    ]:
        fn()
    print("\nAll gradient checks passed.")
