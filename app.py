# ==========================================
# 1. IMPORTS: Saglyat aadhi library import kara
# ==========================================
import matplotlib.pyplot as plt

# ==========================================
# 2. DATA: Ithe tumcha swatahcha data taku shakta
# ==========================================
# Tumche labels (X-axis var kay dakhavaycha te)
categories = ['Python', 'Java', 'C++', 'JavaScript', 'SQL']

# Tumchya values (Y-axis var numbers/scores)
values = [85, 70, 65, 80, 75]

# ==========================================
# 3. SETUP: Figure aani Axis create kara
# ==========================================
fig, ax = plt.subplots(figsize=(9, 5))

# ==========================================
# 4. PLOT: Bar graph banavnya sathi command
# ==========================================
bars = ax.bar(categories, values, color='#3498db', edgecolor='#2980b9', width=0.55, linewidth=1.2)

# ==========================================
# 5. SPINES (Borders): Tumchi main requirement
# ==========================================
# Varchi ani ujeekadchi border kadhun taka
ax.spines['top'].set_visible(False)
ax.spines['right'].set_visible(False)

# Davikadchi ani khalchi border set karne (linewidth ani color)
ax.spines['left'].set(linewidth=1.8, color='#2c3e50')
ax.spines['bottom'].set(linewidth=1.8, color='#2c3e50')

# ==========================================
# 6. LABELS & TITLE: Nav deyne
# ==========================================
ax.set_title("Programming Languages Popularity Graph", fontsize=14, fontweight='bold', color='#2c3e50', pad=15)
ax.set_xlabel("Languages", fontsize=12, fontweight='bold', color='#2c3e50')
ax.set_ylabel("Popularity Score", fontsize=12, fontweight='bold', color='#2c3e50')

# ==========================================
# 7. GRID & VALUES: Background grid ani numbers
# ==========================================
ax.yaxis.grid(True, linestyle='--', alpha=0.6, color='#bdc3c7')
ax.set_axisbelow(True)

# Pratyek bar var varti value print karnyacha loop
for bar in bars:
    height = bar.get_height()
    ax.annotate(f'{height}',
                xy=(bar.get_x() + bar.get_width() / 2, height),
                xytext=(0, 4),  # thode var offset
                textcoords="offset points",
                ha='center', va='bottom', fontsize=10, fontweight='bold', color='#2c3e50')

# ==========================================
# 8. DISPLAY: Layout adjust karun graph dakhavne
# ==========================================
plt.tight_layout()
plt.show()
