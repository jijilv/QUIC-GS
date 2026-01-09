import matplotlib.pyplot as plt

rd_data = [
    (23.14, 1.0),
    (23.10, 7.97),
    (23.09, 8.32),
    (23.09, 8.77),
    (23.09, 9.76),
    (23.09, 11.03),
    (23.09, 12.71),
    (23.09, 15.06),
    (23.08, 16.73),
    (23.08, 17.63),
    (23.05, 18.64),
    (23.02, 20.03),
    (23.00, 21.34),
    (22.95, 23.05),
    (22.90, 24.80),
    (22.81, 27.19),
    (22.69, 29.70),
    (22.59, 31.45),
    (22.44, 33.43),
    (22.15, 36.49),
    (22.00, 42.09),
    (21.80, 44.45),
    (21.44, 48.53),
    (21.15, 49.79),
    (20.98, 51.41),
    (20.93, 58.71),
    (20.76, 60.98),
    (20.59, 63.42),
    (20.39, 66.07),
    (20.17, 68.95),
    (19.95, 72.09),
    (19.70, 75.11),
    (19.45, 78.88),
    (19.15, 83.04),
    (18.83, 87.66),
    (18.52, 92.81),
    (18.17, 98.60),
    (17.77, 114.98),
    (17.40, 123.97),
    (17.00, 134.47),
    (16.59, 146.88),
    (16.13, 161.90),
    (15.65, 180.26),
    (15.54, 244.72),
    (15.04, 288.77),
    (14.48, 352.88),
    (13.79, 453.96),
    (13.39, 529.88),
    (12.91, 601.03),
    (12.94, 637.01),
    (11.76, 1075.18),
]

def plot_rd_curve(rd_data, baseline_psnr=23.14):
    # 去掉 baseline 点 + 截断到 [8, 256]
    filtered = [
        (p, r) for p, r in rd_data
        if r != 1.0 and 8 <= r <= 256
    ]

    filtered = sorted(filtered, key=lambda x: x[1])
    psnr = [p for p, _ in filtered]
    ratio = [r for _, r in filtered]

    plt.figure(figsize=(6, 4))

    # 散点（高级蓝灰）
    plt.scatter(
        ratio,
        psnr,
        marker='x',
        s=42,
        color='#4C72B0'
    )

    # Baseline（柔和红色虚线）
    plt.axhline(
        y=baseline_psnr,
        color='#C44E52',
        linestyle='--',
        linewidth=1.6
    )

    # Baseline 小字标注（在线下方）
    plt.text(
        200,                      # 靠左，不贴边
        baseline_psnr - 0.2,   # 稍微下移
        "Baseline",
        color='#C44E52',
        fontsize=9,
        verticalalignment='top'
    )

    # X 轴
    plt.xscale('log')
    plt.xlim(7, 300)
    xticks = [8, 16, 32, 64, 128, 256]
    plt.xticks(xticks, [str(x) for x in xticks])

    # Y 轴
    yticks = list(range(14, 25, 2))
    plt.ylim(14, 24)
    plt.yticks(yticks, [str(y) for y in yticks])

    plt.xlabel('Compression Ratio (×)')
    plt.ylabel('PSNR (dB)')
    plt.title('R–D Curve (PSNR vs Compression Ratio)')

    # 只画主刻度交叉网格
    plt.grid(
        True,
        which='major',
        axis='both',
        linestyle='-',
        linewidth=0.5,
        alpha=0.35
    )

    plt.tight_layout()
    plt.savefig("pic/rd_curve.pdf", bbox_inches='tight')

    plt.show()


if __name__ == "__main__":
    plot_rd_curve(rd_data)
