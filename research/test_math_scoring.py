from math_scoring import math_equal


def main():
    pairs = [
        ("1/2", r"\frac{1}{2}"),
        ("0.5", r"\frac{1}{2}"),
        ("3", "3.0"),
        (r"(3,\frac{\pi}{2})", r"\left(3,\frac{\pi}{2}\right)"),
    ]
    for prediction, reference in pairs:
        result = math_equal(prediction, reference)
        print(f"{prediction!r} == {reference!r}: {result}")
        assert result


if __name__ == "__main__":
    main()
