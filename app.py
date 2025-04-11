import streamlit as st
import numpy as np
from scipy.stats import nbinom, poisson

# 模型參數
MLB_STD_DEV = 3.8
NB_R = 5.3
NUM_SIMULATIONS = 50000

st.set_page_config(page_title="MLB 星級預測系統", layout="centered")
st.title("🌟 MLB 星級預測系統（上壘率增強版）")
st.markdown("""
加入團隊上壘率（OBP）計算｜雙重結果比較｜三種概率模型
""")

# 星級評分計算
def calculate_stars(prob, mean_diff, over_consistency, volatility, trend_strength):
    if prob >= 50:
        base = 3.0 + (prob - 50) / 12.5
        adjustments = 0.5 * mean_diff + 0.7 * over_consistency + 0.3 * np.log1p(volatility) + 0.4 * trend_strength
    else:
        base = 3.0 - (50 - prob) / 12.5
        adjustments = -0.6 * mean_diff - 0.5 * over_consistency - 0.4 * np.log1p(volatility) - 0.3 * trend_strength
    raw_score = base + adjustments
    return min(5, max(1, round(raw_score, 1)))

# 模擬計算主函式
def simulate_game(home, away, target, use_advanced_data=True):
    def weighted_score(data, opp_allow, opp_pitcher=None, opp_team_stats=None):
        if use_advanced_data and opp_pitcher and opp_team_stats:
            pitcher_factor = (opp_pitcher['era'] * 0.7 + opp_pitcher['baa'] * 0.3) / 4.5
            batting_factor = (data['team_batting'] / 0.250) * 0.6 + (data['team_obp'] / 0.320) * 0.4
            return (
                (data['time_avg'] * 0.35 + data['base_avg'] * 0.25 + data['allow'] * 0.15)
                * (1 + data['over_rate'] * 0.05)
                * (1.2 - pitcher_factor * 0.4)
                * (0.9 + batting_factor * 0.2)
            )
        else:
            return (
                (data['time_avg'] * 0.5 + data['base_avg'] * 0.3 + opp_allow * 0.15)
                * (1 + data['over_rate'] * 0.05)
            )

    home_score = weighted_score(home, away['allow'], away['pitcher'], away) if use_advanced_data else weighted_score(home, away['allow'])
    away_score = weighted_score(away, home['allow'], home['pitcher'], home) if use_advanced_data else weighted_score(away, home['allow'])
    combined_avg = home_score + away_score

    mc_scores = np.random.normal(home_score, MLB_STD_DEV, NUM_SIMULATIONS) + np.random.normal(away_score, MLB_STD_DEV, NUM_SIMULATIONS)
    mc_prob = np.mean(mc_scores > target) * 100

    def generate_nb(mean):
        p = NB_R / (NB_R + mean)
        return nbinom.rvs(NB_R, p, size=NUM_SIMULATIONS)

    def generate_poisson(mean):
        return poisson.rvs(mean, size=NUM_SIMULATIONS)

    nb_prob = np.mean((generate_nb(home_score) + generate_nb(away_score)) > target) * 100
    poisson_prob = np.mean((generate_poisson(home_score) + generate_poisson(away_score)) > target) * 100
    final_prob = mc_prob * 0.5 + nb_prob * 0.3 + poisson_prob * 0.2

    mean_diff = (combined_avg - target) / 3
    over_consistency = home['over_rate'] + away['over_rate'] - 1
    volatility = (np.std([home['time_avg'], home['base_avg']]) + np.std([away['time_avg'], away['base_avg']])) / 2
    trend_strength = ((home['time_avg'] - home['base_avg']) + (away['time_avg'] - away['base_avg'])) / 5
    stars = calculate_stars(final_prob, mean_diff, over_consistency, volatility, trend_strength)

    return {
        'home_score': home_score,
        'away_score': away_score,
        'combined_avg': combined_avg,
        'mc_prob': mc_prob,
        'nb_prob': nb_prob,
        'poisson_prob': poisson_prob,
        'probability': final_prob,
        'stars': stars,
        'recommendation': '大分' if final_prob >= 50 else '小分',
        'type': '含進階數據' if use_advanced_data else '不含進階數據'
    }

# 表單輸入
with st.form("mlb_form"):
    st.subheader("比賽基本資料")
    home_team = st.text_input("主隊名稱")
    away_team = st.text_input("客隊名稱")
    target = st.number_input("盤口總分", step=0.5)
    
    col1, col2 = st.columns(2)
    with col1:
        st.markdown(f"#### {home_team or '主隊'} 數據")
        h_time_avg = st.number_input("時段場均得分", key="h1")
        h_base_avg = st.number_input("整體場均得分", key="h2")
        h_allow = st.number_input("場均失分", key="h3")
        h_over = st.slider("大分過盤率 (%)", 0, 100, 50, key="h4") / 100
        h_avg = st.number_input("團隊打擊率 (如 0.250)", key="h5")
        h_obp = st.number_input("團隊上壘率 (如 0.320)", key="h6")
        h_era = st.number_input("投手 ERA", key="h7")
        h_baa = st.number_input("投手 BAA", key="h8")
    with col2:
        st.markdown(f"#### {away_team or '客隊'} 數據")
        a_time_avg = st.number_input("時段場均得分", key="a1")
        a_base_avg = st.number_input("整體場均得分", key="a2")
        a_allow = st.number_input("場均失分", key="a3")
        a_over = st.slider("大分過盤率 (%)", 0, 100, 50, key="a4") / 100
        a_avg = st.number_input("團隊打擊率 (如 0.250)", key="a5")
        a_obp = st.number_input("團隊上壘率 (如 0.320)", key="a6")
        a_era = st.number_input("投手 ERA", key="a7")
        a_baa = st.number_input("投手 BAA", key="a8")

    submitted = st.form_submit_button("開始分析")

# 執行模擬與結果顯示
if submitted:
    home = {
        'team_name': home_team, 'time_avg': h_time_avg, 'base_avg': h_base_avg,
        'allow': h_allow, 'over_rate': h_over, 'team_batting': h_avg, 'team_obp': h_obp,
        'pitcher': {'era': h_era, 'baa': h_baa}
    }
    away = {
        'team_name': away_team, 'time_avg': a_time_avg, 'base_avg': a_base_avg,
        'allow': a_allow, 'over_rate': a_over, 'team_batting': a_avg, 'team_obp': a_obp,
        'pitcher': {'era': a_era, 'baa': a_baa}
    }

    adv = simulate_game(home, away, target, use_advanced_data=True)
    base = simulate_game(home, away, target, use_advanced_data=False)

    def display(result):
        st.subheader(f"{result['type']} 結果")
        st.write(f"預期總分：{result['combined_avg']:.2f}，推薦：**{result['recommendation']}**")
        st.write(f"蒙地卡羅：{result['mc_prob']:.1f}%｜負二項：{result['nb_prob']:.1f}%｜泊松：{result['poisson_prob']:.1f}%")
        st.write(f"綜合機率：**{result['probability']:.1f}%**｜星級：{'★' * int(result['stars']) + '☆' * (5 - int(result['stars']))} ({result['stars']}/5.0)")

    display(adv)
    display(base)
