def edit_distance(a, b):
    """أقل عدد عمليات (استبدال/حذف/إضافة) لتحويل a إلى b.
    a و b ممكن يكونوا نصوص (للحروف) أو lists من كلمات."""
    dp = [[0] * (len(b) + 1) for _ in range(len(a) + 1)]
    for i in range(len(a) + 1):
        dp[i][0] = i
    for j in range(len(b) + 1):
        dp[0][j] = j
    for i in range(1, len(a) + 1):
        for j in range(1, len(b) + 1):
            cost = 0 if a[i - 1] == b[j - 1] else 1
            dp[i][j] = min(
                dp[i - 1][j] + 1,         # حذف
                dp[i][j - 1] + 1,         # إضافة
                dp[i - 1][j - 1] + cost,  # استبدال (أو تطابق)
            )
    return dp[len(a)][len(b)]


def cer(truth, pred):
    """Character Error Rate لجملة واحدة."""
    if len(truth) == 0:
        return 0.0 if len(pred) == 0 else 1.0
    return edit_distance(truth, pred) / len(truth)


def wer(truth, pred):
    """Word Error Rate: نفس الفكرة لكن على الكلمات."""
    t, p = truth.split(), pred.split()
    if len(t) == 0:
        return 0.0 if len(p) == 0 else 1.0
    return edit_distance(t, p) / len(t)


def corpus_cer(truths, preds):
    """CER على مجموعة جمل: مجموع الأخطاء ÷ مجموع الحروف.
    ده الأدق للتقرير (مش متوسط CER كل جملة)."""
    errors = sum(edit_distance(t, p) for t, p in zip(truths, preds))
    total = sum(len(t) for t in truths)
    return errors / total


if __name__ == "__main__":
    print(cer("hello", "hallo"))                    # المتوقع 0.2
    print(cer("hello", "hello"))                    # المتوقع 0.0
    print(cer("hello", "helo"))                     # المتوقع 0.2 (حرف ناقص)
    print(wer("the cat sat", "the dog sat"))        # المتوقع 0.333...