"""
販売予測アプリ
Streamlitを使用した時系列販売予測ツール
"""
import streamlit as st
import pandas as pd
import numpy as np
from datetime import timedelta, date
import plotly.graph_objects as go
import warnings
warnings.filterwarnings('ignore')

# ページ設定（最初に実行する必要がある）
st.set_page_config(
    page_title="販売予測アプリ",
    page_icon="📈",
    layout="wide",
    initial_sidebar_state="expanded"
)

# 予測モデルのインポート
try:
    from prophet import Prophet
    PROPHET_AVAILABLE = True
except ImportError:
    PROPHET_AVAILABLE = False
    st.warning("Prophetがインストールされていません。pip install prophet でインストールしてください。")

try:
    from pmdarima import auto_arima
    ARIMA_AVAILABLE = True
except ImportError:
    ARIMA_AVAILABLE = False
    st.warning("pmdarimaがインストールされていません。pip install pmdarima でインストールしてください。")

from sklearn.linear_model import LinearRegression

# タイトル
st.title("📈 販売予測アプリ")
st.markdown("---")

# サイドバーの幅を調整（CSSで広げる）
st.markdown(
    """
    <style>
    [data-testid="stSidebar"] {
        min-width: 420px !important;
        max-width: 420px !important;
    }
    [data-testid="stSidebar"] > div:first-child {
        width: 420px !important;
    }
    [data-testid="stSidebar"] [data-testid="stVerticalBlock"] {
        width: 420px !important;
    }
    </style>
    """,
    unsafe_allow_html=True,
)

# サイドバー：設定
st.sidebar.header("基本設定")

# CSVファイルのアップロード
uploaded_file = st.sidebar.file_uploader(
    "CSVファイルをアップロード",
    type=['csv'],
    help="売上明細CSVファイルをアップロードしてください。昨日までのデータを含むファイルをアップロードしてください。"
)

@st.cache_data
def load_data(file_path_or_buffer):
    """CSVファイルを読み込んで前処理"""
    try:
        # CSVファイルを読み込み（ファイルパスまたはファイルオブジェクトに対応）
        df = pd.read_csv(file_path_or_buffer, encoding='utf-8-sig')
        
        # 日付列の変換（「2025年12月01日」形式 → datetime）
        def parse_date(date_str):
            """日付文字列をdatetimeに変換"""
            try:
                # 「2025年12月01日」形式を処理
                date_str = str(date_str).replace('年', '-').replace('月', '-').replace('日', '')
                return pd.to_datetime(date_str, format='%Y-%m-%d')
            except:
                return pd.NaT
        
        df['売上日付_datetime'] = df['売上日付'].apply(parse_date)
        
        # 税抜売上金額の処理（カンマ区切りを数値に変換）
        df['税抜売上金額_数値'] = df['税抜売上金額'].astype(str).str.replace(',', '').astype(float)
        
        # 商品コードと商品名の組み合わせを作成
        if '商品コード' in df.columns and '商品名' in df.columns:
            df['商品コード_商品名'] = df['商品コード'].astype(str).str.strip() + ' - ' + df['商品名'].astype(str).str.strip()
        
        # 部門コードと部門名の組み合わせを作成
        if '部門コード' in df.columns and '部門名' in df.columns:
            df['部門コード_部門名'] = df['部門コード'].astype(str).str.strip() + ' - ' + df['部門名'].astype(str).str.strip()
        
        # 受注方法コードと受注方法名の組み合わせを作成
        if '受注方法コード' in df.columns and '受注方法名' in df.columns:
            df['受注方法コード_受注方法名'] = df['受注方法コード'].astype(str).str.strip() + ' - ' + df['受注方法名'].astype(str).str.strip()
        
        # 日付ごとに集計（全商品）
        daily_data_all = df.groupby('売上日付_datetime').agg({
            '売上数量': 'sum',
            '税抜売上金額_数値': 'sum'
        }).reset_index()
        
        daily_data_all.columns = ['日付', '売上数量', '税抜売上金額']
        daily_data_all = daily_data_all.sort_values('日付').reset_index(drop=True)
        
        # 欠損日（売上が0の日）を補完
        if len(daily_data_all) > 0:
            min_date = daily_data_all['日付'].min()
            max_date = daily_data_all['日付'].max()
            # 全ての日付範囲を作成
            all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
            # 日付データフレームを作成
            date_df = pd.DataFrame({'日付': all_dates})
            # 既存データとマージ（欠損日は0で埋める）
            daily_data_all = date_df.merge(daily_data_all, on='日付', how='left')
            daily_data_all['売上数量'] = daily_data_all['売上数量'].fillna(0)
            daily_data_all['税抜売上金額'] = daily_data_all['税抜売上金額'].fillna(0)
            daily_data_all = daily_data_all.sort_values('日付').reset_index(drop=True)
        
        return daily_data_all, df
    except Exception as e:
        st.error(f"データの読み込みエラー: {str(e)}")
        return None, None

def sort_by_code(item_list, is_first_item_all=True):
    """コードでソート（数値として処理）"""
    if is_first_item_all:
        # 最初の要素（"全ての..."）を除く
        all_item = item_list[0] if item_list else ""
        code_items = item_list[1:] if len(item_list) > 1 else []
    else:
        all_item = ""
        code_items = item_list
    
    def extract_code(item):
        """コードを抽出して数値に変換"""
        try:
            code_str = item.split(' - ')[0].strip()
            # 数値に変換できるか試す
            return int(float(code_str))
        except (ValueError, IndexError):
            # 数値に変換できない場合は0を返して後ろに配置
            return 0
    
    # コードでソート
    sorted_code_items = sorted(code_items, key=extract_code)
    
    if is_first_item_all:
        return [all_item] + sorted_code_items
    else:
        return sorted_code_items

def filter_by_department(df, selected_departments):
    """選択された部門でデータをフィルタリング"""
    if not selected_departments or len(selected_departments) == 0 or "全ての部門" in selected_departments:
        return df
    else:
        # 選択された部門のコードを取得
        department_codes = [dept.split(' - ')[0] for dept in selected_departments]
        filtered_df = df[df['部門コード'].astype(str).str.strip().isin(department_codes)]
        return filtered_df

def filter_by_order_method(df, selected_order_methods):
    """選択された受注方法でデータをフィルタリング"""
    if not selected_order_methods or len(selected_order_methods) == 0 or "全ての受注方法" in selected_order_methods:
        return df
    else:
        # 選択された受注方法のコードを取得
        order_method_codes = [method.split(' - ')[0] for method in selected_order_methods]
        filtered_df = df[df['受注方法コード'].astype(str).str.strip().isin(order_method_codes)]
        return filtered_df

def filter_by_product(df, selected_products):
    """選択された商品でデータをフィルタリング"""
    if not selected_products or len(selected_products) == 0 or "全ての商品" in selected_products:
        # 全商品の集計
        daily_data = df.groupby('売上日付_datetime').agg({
            '売上数量': 'sum',
            '税抜売上金額_数値': 'sum'
        }).reset_index()
    else:
        # 選択された商品のコードを取得
        product_codes = [prod.split(' - ')[0] for prod in selected_products]
        filtered_df = df[df['商品コード'].astype(str).str.strip().isin(product_codes)]
        daily_data = filtered_df.groupby('売上日付_datetime').agg({
            '売上数量': 'sum',
            '税抜売上金額_数値': 'sum'
        }).reset_index()
    
    daily_data.columns = ['日付', '売上数量', '税抜売上金額']
    daily_data = daily_data.sort_values('日付').reset_index(drop=True)
    
    # 欠損日（売上が0の日）を補完
    if len(daily_data) > 0:
        min_date = daily_data['日付'].min()
        max_date = daily_data['日付'].max()
        # 全ての日付範囲を作成
        all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
        # 日付データフレームを作成
        date_df = pd.DataFrame({'日付': all_dates})
        # 既存データとマージ（欠損日は0で埋める）
        daily_data = date_df.merge(daily_data, on='日付', how='left')
        daily_data['売上数量'] = daily_data['売上数量'].fillna(0)
        daily_data['税抜売上金額'] = daily_data['税抜売上金額'].fillna(0)
        daily_data = daily_data.sort_values('日付').reset_index(drop=True)
    
    return daily_data

# データ読み込み
if uploaded_file is not None:
    daily_data, raw_data = load_data(uploaded_file)
else:
    daily_data, raw_data = None, None

if daily_data is not None and len(daily_data) > 0:
    # サイドバー：設定項目
    # 部門選択
    st.sidebar.markdown("### 部門選択")
    if '部門コード_部門名' in raw_data.columns:
        # 部門リストを作成（重複を削除、コードでソート）
        department_unique = raw_data['部門コード_部門名'].unique().tolist()
        department_list = sort_by_code(["全ての部門"] + department_unique)
        selected_departments = st.sidebar.multiselect(
            "予測対象の部門（複数選択可）",
            department_list,
            default=["全ての部門"],
            help="特定の部門を選択すると、その部門のみの予測を行います。複数選択可能です。"
        )
        # 選択された部門でデータをフィルタリング
        raw_data_filtered = filter_by_department(raw_data, selected_departments)
    else:
        selected_departments = ["全ての部門"]
        raw_data_filtered = raw_data
        st.sidebar.info("部門コード・部門名が見つかりません。")
    
    # 受注方法選択
    st.sidebar.markdown("### 受注方法選択")
    if '受注方法コード_受注方法名' in raw_data_filtered.columns:
        # 受注方法リストを作成（重複を削除、コードでソート）
        order_method_unique = raw_data_filtered['受注方法コード_受注方法名'].unique().tolist()
        order_method_list = sort_by_code(["全ての受注方法"] + order_method_unique)
        selected_order_methods = st.sidebar.multiselect(
            "予測対象の受注方法（複数選択可）",
            order_method_list,
            default=["全ての受注方法"],
            help="特定の受注方法を選択すると、その受注方法のみの予測を行います。複数選択可能です。"
        )
        # 選択された受注方法でデータをフィルタリング
        raw_data_filtered = filter_by_order_method(raw_data_filtered, selected_order_methods)
    else:
        selected_order_methods = ["全ての受注方法"]
        st.sidebar.info("受注方法コード・受注方法名が見つかりません。")
    
    # 商品選択
    st.sidebar.markdown("### 商品選択")
    if '商品コード_商品名' in raw_data_filtered.columns:
        # 商品リストを作成（重複を削除、コードでソート）
        product_unique = raw_data_filtered['商品コード_商品名'].unique().tolist()
        product_list = sort_by_code(["全ての商品"] + product_unique)
        selected_products = st.sidebar.multiselect(
            "予測対象の商品（複数選択可）",
            product_list,
            default=["全ての商品"],
            help="特定の商品を選択すると、その商品のみの予測を行います。複数選択可能です。"
        )
        # 選択された商品でデータをフィルタリング（部門・受注方法フィルタリング後のデータを使用）
        daily_data_filtered = filter_by_product(raw_data_filtered, selected_products)
        # フィルタリング後のデータが空でないか確認
        if len(daily_data_filtered) > 0:
            daily_data = daily_data_filtered
        else:
            st.sidebar.warning("選択された条件にデータがありません。")
            daily_data = None
    else:
        selected_products = ["全ての商品"]
        # 商品がない場合は部門・受注方法のみで集計
        if not selected_departments or len(selected_departments) == 0 or "全ての部門" in selected_departments:
            daily_data = raw_data_filtered.groupby('売上日付_datetime').agg({
                '売上数量': 'sum',
                '税抜売上金額_数値': 'sum'
            }).reset_index()
            daily_data.columns = ['日付', '売上数量', '税抜売上金額']
            daily_data = daily_data.sort_values('日付').reset_index(drop=True)
            
            # 欠損日（売上が0の日）を補完
            if len(daily_data) > 0:
                min_date = daily_data['日付'].min()
                max_date = daily_data['日付'].max()
                # 全ての日付範囲を作成
                all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
                # 日付データフレームを作成
                date_df = pd.DataFrame({'日付': all_dates})
                # 既存データとマージ（欠損日は0で埋める）
                daily_data = date_df.merge(daily_data, on='日付', how='left')
                daily_data['売上数量'] = daily_data['売上数量'].fillna(0)
                daily_data['税抜売上金額'] = daily_data['税抜売上金額'].fillna(0)
                daily_data = daily_data.sort_values('日付').reset_index(drop=True)
        else:
            daily_data = filter_by_product(raw_data_filtered, selected_products)
        st.sidebar.info("商品コード・商品名が見つかりません。")
    
    # データの基本情報を表示（フィルタリング後）
    if daily_data is not None and len(daily_data) > 0:
        # 予測モデルの選択
        st.sidebar.markdown("### 予測モデル")
        model_options = []
        if PROPHET_AVAILABLE:
            model_options.append("Prophet")
        if ARIMA_AVAILABLE:
            model_options.append("ARIMA")
        model_options.append("Linear Regression")
        
        selected_model = st.sidebar.selectbox(
            "モデルを選択",
            model_options,
            index=0
        )
        
        # 選択されたモデルの特徴を表示
        with st.sidebar.expander("📖 モデルの特徴"):
            if "Prophet" in selected_model:
                st.markdown("""
                **Prophet（時系列予測モデル）**
                
                - **特徴**: Facebookが開発した時系列予測モデル
                - **強み**: 
                  - 週次・日次・年次の季節性を自動検出
                  - トレンドと季節性を分離して分析
                  - 休日やイベントの影響を考慮可能
                  - 欠損値や外れ値に強い
                - **適用場面**: 
                  - 週次パターンが明確なデータ
                  - 季節性があるデータ
                  - 長期的なトレンド予測
                  - 去年の同じ時期のパターンを予測に反映したい場合
                - **計算時間**: 中程度
                - **注意事項**: 
                  - 年次季節性を計算するため、データ期間が1年以上あることが推奨されます
                  - データ期間が1年未満の場合、年次季節性は正しく学習されない可能性があります
                  - 計算時間はデータ量に応じて長くなる場合があります
                """)
            elif "ARIMA" in selected_model:
                st.markdown("""
                **ARIMA（自己回帰移動平均モデル）**
                
                - **特徴**: 統計的手法に基づく時系列予測モデル
                - **強み**: 
                  - 統計的に堅牢で信頼性が高い
                  - パラメータが自動最適化される（Auto ARIMA）
                  - トレンドと季節性を考慮
                  - 短期予測に適している
                - **適用場面**: 
                  - 統計的に安定したデータ
                  - 短期間の予測
                  - トレンドが明確なデータ
                - **計算時間**: やや長め
                """)
            else:
                st.markdown("""
                **Linear Regression（線形回帰）**
                
                - **特徴**: シンプルな線形回帰モデル
                - **強み**: 
                  - 理解しやすく、解釈が容易
                  - 計算が高速
                  - トレンドを直線的に予測
                  - データ量が少なくても動作
                - **適用場面**: 
                  - シンプルなトレンド予測
                  - データ量が少ない場合
                  - 迅速な予測が必要な場合
                - **計算時間**: 非常に高速
                """)
        
        # データの最終日を選択
        st.sidebar.markdown("### データの最終日設定")
        
        # 元のデータの最終日を取得
        original_last_date = daily_data['日付'].max()
        original_first_date = daily_data['日付'].min()
        if isinstance(original_last_date, pd.Timestamp):
            original_last_date_date = original_last_date.date()
            original_first_date_date = original_first_date.date()
        else:
            original_last_date_date = original_last_date
            original_first_date_date = original_first_date
        
        # 昨日の日付を取得（デフォルト値として使用）
        yesterday = date.today() - timedelta(days=1)
        
        # データの最終日を選択（元のデータの最終日より後の日付も選択可能）
        max_selectable_date = original_last_date_date + timedelta(days=365)
        
        # デフォルト値を決定（昨日が範囲内にある場合は昨日、そうでない場合は元のデータの最終日）
        if original_first_date_date <= yesterday <= max_selectable_date:
            default_value = yesterday
        else:
            default_value = original_last_date_date
        
        data_end_date = st.sidebar.date_input(
            "予測に使用するデータの最終日",
            value=default_value,
            min_value=original_first_date_date,
            max_value=max_selectable_date,
            help="この日付までのデータを使用して予測を行います。元のデータの最終日より後の日付を選択した場合、その期間のデータがない日は0として扱われます。"
        )
        
        # 元のデータの最小日付を取得
        original_min_date = daily_data['日付'].min()
        if isinstance(original_min_date, pd.Timestamp):
            original_min_date_date = original_min_date.date()
        else:
            original_min_date_date = original_min_date
        
        # 選択した最終日までのデータでフィルタリング（元のデータの範囲内）
        daily_data_for_forecast = daily_data[daily_data['日付'] <= pd.Timestamp(original_last_date_date)].copy()
        
        # 選択した最終日までの欠損日を0で補完
        # 元のデータの最小日付から選択した最終日まで全ての日付を作成
        if len(daily_data_for_forecast) > 0:
            min_date = original_min_date_date
            max_date = pd.Timestamp(data_end_date)
            # 全ての日付範囲を作成
            all_dates = pd.date_range(start=min_date, end=max_date, freq='D')
            # 日付データフレームを作成
            date_df = pd.DataFrame({'日付': all_dates})
            # 既存データとマージ（欠損日は0で埋める）
            daily_data_for_forecast = date_df.merge(daily_data_for_forecast, on='日付', how='left')
            daily_data_for_forecast['売上数量'] = daily_data_for_forecast['売上数量'].fillna(0)
            daily_data_for_forecast['税抜売上金額'] = daily_data_for_forecast['税抜売上金額'].fillna(0)
            daily_data_for_forecast = daily_data_for_forecast.sort_values('日付').reset_index(drop=True)
        
        # 予測期間の設定
        st.sidebar.markdown("### 予測期間")
        
        # 予測開始日の設定方法を選択
        forecast_start_method = st.sidebar.radio(
            "予測開始日の設定方法",
            ["データ最終日から○日後", "日付で指定"],
            index=0
        )
        
        if forecast_start_method == "データ最終日から○日後":
            # データ最終日から何日後から予測を開始するか
            days_after_last = st.sidebar.number_input(
                "データ最終日から何日後から予測を開始",
                min_value=0,
                max_value=365,
                value=1,
                step=1,
                help="0を選択するとデータ最終日の翌日から予測を開始します"
            )
            start_date = data_end_date + timedelta(days=days_after_last)
        else:
            # 日付で直接指定
            min_date = data_end_date + timedelta(days=1)
            max_date = min_date + timedelta(days=365)
            start_date = st.sidebar.date_input(
                "開始日",
                value=min_date,
                min_value=min_date,
                max_value=max_date
            )
        
        # 予測終了日を設定
        end_date = st.sidebar.date_input(
            "終了日",
            value=start_date + timedelta(days=30),
            min_value=start_date,
            max_value=start_date + timedelta(days=365)
        )
        
        # 予測日数を計算
        forecast_days = (end_date - start_date).days + 1
        
        # 予測期間の情報を表示
        st.sidebar.markdown(f"**予測開始日**: {start_date.strftime('%Y年%m月%d日')}")
        st.sidebar.markdown(f"**予測終了日**: {end_date.strftime('%Y年%m月%d日')}")
        st.sidebar.markdown(f"**予測日数**: {forecast_days}日")
        
        # 予測実行ボタン
        st.sidebar.markdown("---")
        run_forecast = st.sidebar.button("予測を実行", type="primary", use_container_width=True)
        
        # メインコンテンツ
        # データ情報とヘルプ情報を表示
        col_info1, col_info2 = st.columns([1, 1])
        with col_info1:
            st.markdown("### データ情報")
            min_data_date = daily_data['日付'].min()
            max_data_date = daily_data['日付'].max()
            st.write(f"**元データ期間**: {min_data_date.strftime('%Y年%m月%d日')} ～ {max_data_date.strftime('%Y年%m月%d日')}")
            data_days = len(daily_data)
            st.write(f"**元データ件数**: {data_days}日")
            st.write(f"**元データ最終売上日**: {max_data_date.strftime('%Y年%m月%d日')}")
            
            # 予測に使用するデータの情報
            if 'daily_data_for_forecast' in locals():
                forecast_min_date = daily_data_for_forecast['日付'].min()
                forecast_max_date = daily_data_for_forecast['日付'].max()
                forecast_data_days = len(daily_data_for_forecast)
                st.write(f"**予測使用データ期間**: {forecast_min_date.strftime('%Y年%m月%d日')} ～ {forecast_max_date.strftime('%Y年%m月%d日')}")
                st.write(f"**予測使用データ件数**: {forecast_data_days}日")
                st.write(f"**予測使用データ最終日**: {forecast_max_date.strftime('%Y年%m月%d日')}")
            
            if selected_departments and len(selected_departments) > 0 and "全ての部門" not in selected_departments:
                dept_display = ", ".join(selected_departments) if len(selected_departments) <= 3 else f"{len(selected_departments)}件選択"
                st.write(f"**選択部門**: {dept_display}")
            if 'selected_order_methods' in locals() and selected_order_methods and len(selected_order_methods) > 0 and "全ての受注方法" not in selected_order_methods:
                method_display = ", ".join(selected_order_methods) if len(selected_order_methods) <= 3 else f"{len(selected_order_methods)}件選択"
                st.write(f"**選択受注方法**: {method_display}")
            if selected_products and len(selected_products) > 0 and "全ての商品" not in selected_products:
                prod_display = ", ".join(selected_products) if len(selected_products) <= 3 else f"{len(selected_products)}件選択"
                st.write(f"**選択商品**: {prod_display}")
        with col_info2:
            st.markdown("### 📊 現在のデータ量から期待できる予測精度")
            
            # 予測に使用するデータ件数を取得
            if 'daily_data_for_forecast' in locals():
                forecast_data_days = len(daily_data_for_forecast)
            else:
                forecast_data_days = data_days
            
            # データ件数に基づいて予測精度を判定
            if forecast_data_days < 30:
                accuracy_level = "⚠️ データ不足"
                accuracy_desc = "30日未満のデータでは、基本的な予測は難しい可能性があります。より多くのデータを取得することを推奨します。"
                accuracy_color = "red"
            elif forecast_data_days < 60:
                accuracy_level = "📈 基本的な予測が可能"
                accuracy_desc = "基本的なトレンドの予測が可能です。週次パターンの検出にはさらにデータが必要です。"
                accuracy_color = "orange"
            elif forecast_data_days < 90:
                accuracy_level = "📊 週次パターンの検出が可能"
                accuracy_desc = "週次パターンを考慮した予測が可能です。季節性の検出にはさらにデータが必要です。"
                accuracy_color = "blue"
            elif forecast_data_days < 180:
                accuracy_level = "✅ 季節性の検出が可能（推奨）"
                accuracy_desc = "週次パターンや季節性を考慮した予測が可能です。一般的なビジネス予測に適したデータ量です。"
                accuracy_color = "green"
            else:
                accuracy_level = "🌟 より高精度な予測が可能"
                accuracy_desc = "長期トレンド、季節性、週次パターンを総合的に考慮した高精度な予測が可能です。"
                accuracy_color = "green"
            
            st.markdown(f"**{accuracy_level}**")
            st.markdown(f"<span style='color: {accuracy_color}'>{accuracy_desc}</span>", unsafe_allow_html=True)
            
            # 参照情報
            with st.expander("📚 データ量と精度の関係について"):
                st.markdown("""
                - **30日以上**: 基本的な予測が可能
                - **60日以上**: 週次パターンの検出が可能
                - **90日以上**: 季節性の検出が可能（推奨）
                - **180日以上**: より高精度な予測が可能
                
                データ期間が長いほど、予測モデルはトレンドや季節性をより正確に学習できます。
                """)
        
        st.markdown("---")
        
        # 予測結果の初期化
        forecast_quantity = None
        forecast_amount = None
        forecast_dates_quantity = pd.date_range(start=start_date, end=end_date, freq='D')
        forecast_dates_amount = pd.date_range(start=start_date, end=end_date, freq='D')
        
        # 予測モデルの実行
        if run_forecast:
            with st.spinner("予測を実行中..."):
                
                # データの最後の日付から予測終了日までの日数を計算
                last_data_date = daily_data['日付'].max()
                # 日付の型を統一（Timestampまたはdate）
                if isinstance(last_data_date, pd.Timestamp):
                    last_data_date_date = last_data_date.date()
                else:
                    last_data_date_date = last_data_date
                days_to_forecast_end = (end_date - last_data_date_date).days
                
                # 予測期間が負の値の場合は、予測開始日から予測終了日までの日数を使用
                if days_to_forecast_end < 0:
                    days_to_forecast_end = forecast_days
                
                # 売上数量の予測（選択した最終日までのデータを使用）
                quantity_series = daily_data_for_forecast[['日付', '売上数量']].copy()
                quantity_series.columns = ['ds', 'y']
                
                if "Prophet" in selected_model and PROPHET_AVAILABLE:
                    # Prophetモデル
                    # 年次季節性を常に有効化（データ期間が1年以上ある場合のみ有効）
                    # データ期間が1年未満でも設定は可能だが、正しく学習されない可能性がある
                    model_q = Prophet(
                        daily_seasonality=True,
                        weekly_seasonality=True,
                        yearly_seasonality=True,  # 年次季節性を常に有効化
                        seasonality_mode='multiplicative'
                    )
                    model_q.fit(quantity_series)
                    # データの最後の日付から予測終了日までの日数で予測
                    future_q = model_q.make_future_dataframe(periods=days_to_forecast_end)
                    forecast_q = model_q.predict(future_q)
                    # 予測開始日以降のデータを取得し、必要な日数分に調整
                    forecast_filtered = forecast_q[forecast_q['ds'] >= pd.Timestamp(start_date)]['yhat'].values
                    forecast_quantity = forecast_filtered[:forecast_days] if len(forecast_filtered) >= forecast_days else forecast_filtered
                    # 負の値を0にクリップ（売上数量は負にならない）
                    forecast_quantity = np.maximum(forecast_quantity, 0)
                    
                elif "ARIMA" in selected_model and ARIMA_AVAILABLE:
                    # ARIMAモデル
                    # m=365は計算が重すぎるため、週次季節性のみを使用
                    # パラメータの範囲を制限して高速化
                    model_q = auto_arima(
                        quantity_series['y'],
                        seasonal=True,
                        m=7,  # 週次季節性のみ（m=365は計算が重すぎる）
                        stepwise=True,
                        suppress_warnings=True,
                        max_p=5,
                        max_q=5,
                        max_d=2,
                        max_P=2,
                        max_Q=2,
                        max_D=1,
                        start_p=0,
                        start_q=0,
                        start_P=0,
                        start_Q=0,
                        n_jobs=-1  # 並列処理を有効化
                    )
                    forecast_quantity = model_q.predict(n_periods=forecast_days)
                    # 負の値を0にクリップ（売上数量は負にならない）
                    forecast_quantity = np.maximum(forecast_quantity, 0)
                    
                else:
                    # Linear Regression
                    X = np.arange(len(quantity_series)).reshape(-1, 1)
                    y = quantity_series['y'].values
                    model_q = LinearRegression()
                    model_q.fit(X, y)
                    X_future = np.arange(len(quantity_series), len(quantity_series) + forecast_days).reshape(-1, 1)
                    forecast_quantity = model_q.predict(X_future)
                    # 負の値を0にクリップ（売上数量は負にならない）
                    forecast_quantity = np.maximum(forecast_quantity, 0)
                
                # 税抜売上金額の予測（選択した最終日までのデータを使用）
                amount_series = daily_data_for_forecast[['日付', '税抜売上金額']].copy()
                amount_series.columns = ['ds', 'y']
                
                if "Prophet" in selected_model and PROPHET_AVAILABLE:
                    # Prophetモデル
                    # 年次季節性を常に有効化（データ期間が1年以上ある場合のみ有効）
                    # データ期間が1年未満でも設定は可能だが、正しく学習されない可能性がある
                    model_a = Prophet(
                        daily_seasonality=True,
                        weekly_seasonality=True,
                        yearly_seasonality=True,  # 年次季節性を常に有効化
                        seasonality_mode='multiplicative'
                    )
                    model_a.fit(amount_series)
                    # データの最後の日付から予測終了日までの日数で予測
                    future_a = model_a.make_future_dataframe(periods=days_to_forecast_end)
                    forecast_a = model_a.predict(future_a)
                    # 予測開始日以降のデータを取得し、必要な日数分に調整
                    forecast_filtered = forecast_a[forecast_a['ds'] >= pd.Timestamp(start_date)]['yhat'].values
                    forecast_amount = forecast_filtered[:forecast_days] if len(forecast_filtered) >= forecast_days else forecast_filtered
                    # 負の値を0にクリップ（売上金額は負にならない）
                    forecast_amount = np.maximum(forecast_amount, 0)
                    
                elif "ARIMA" in selected_model and ARIMA_AVAILABLE:
                    # ARIMAモデル
                    # m=365は計算が重すぎるため、週次季節性のみを使用
                    # パラメータの範囲を制限して高速化
                    model_a = auto_arima(
                        amount_series['y'],
                        seasonal=True,
                        m=7,  # 週次季節性のみ（m=365は計算が重すぎる）
                        stepwise=True,
                        suppress_warnings=True,
                        max_p=5,
                        max_q=5,
                        max_d=2,
                        max_P=2,
                        max_Q=2,
                        max_D=1,
                        start_p=0,
                        start_q=0,
                        start_P=0,
                        start_Q=0,
                        n_jobs=-1  # 並列処理を有効化
                    )
                    forecast_amount = model_a.predict(n_periods=forecast_days)
                    # 負の値を0にクリップ（売上金額は負にならない）
                    forecast_amount = np.maximum(forecast_amount, 0)
                    
                else:
                    # Linear Regression
                    X = np.arange(len(amount_series)).reshape(-1, 1)
                    y = amount_series['y'].values
                    model_a = LinearRegression()
                    model_a.fit(X, y)
                    X_future = np.arange(len(amount_series), len(amount_series) + forecast_days).reshape(-1, 1)
                    forecast_amount = model_a.predict(X_future)
                    # 負の値を0にクリップ（売上金額は負にならない）
                    forecast_amount = np.maximum(forecast_amount, 0)
        
        # グラフの表示
        if forecast_quantity is not None and forecast_amount is not None:
            # タイトルの準備
            department_title = ""
            if selected_departments and len(selected_departments) > 0 and "全ての部門" not in selected_departments:
                if len(selected_departments) == 1:
                    department_title = f"【{selected_departments[0]}】"
                else:
                    department_title = f"【部門{len(selected_departments)}件】"
            
            order_method_title = ""
            if 'selected_order_methods' in locals() and selected_order_methods and len(selected_order_methods) > 0 and "全ての受注方法" not in selected_order_methods:
                if len(selected_order_methods) == 1:
                    order_method_title = f"【{selected_order_methods[0]}】"
                else:
                    order_method_title = f"【受注方法{len(selected_order_methods)}件】"
            
            product_title = ""
            if selected_products and len(selected_products) > 0 and "全ての商品" not in selected_products:
                if len(selected_products) == 1:
                    product_title = f"【{selected_products[0]}】"
                else:
                    product_title = f"【商品{len(selected_products)}件】"
            
            filter_title = (department_title + order_method_title + product_title).strip() if (department_title + order_method_title + product_title).strip() else ""
            
            # 売上数量のグラフ
            st.subheader("売上数量の予測")
            fig_quantity = go.Figure()
            
            # 実績データ（予測に使用したデータを表示）
            fig_quantity.add_trace(go.Scatter(
                x=daily_data_for_forecast['日付'],
                y=daily_data_for_forecast['売上数量'],
                mode='lines+markers',
                name='実績',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=6)
            ))
            
            # 予測データ
            fig_quantity.add_trace(go.Scatter(
                x=forecast_dates_quantity,
                y=forecast_quantity,
                mode='lines+markers',
                name='予測',
                line=dict(color='#ff7f0e', width=2, dash='dash'),
                marker=dict(size=6)
            ))
            
            fig_quantity.update_layout(
                title=f'{filter_title}売上数量の推移と予測（{selected_model}）',
                xaxis_title='日付',
                yaxis_title='売上数量',
                hovermode='x unified',
                height=400,
                template='plotly_white'
            )
            
            st.plotly_chart(fig_quantity, use_container_width=True)
            
            # 予測期間のサマリー（グラフの下に表示）
            st.markdown("---")
            st.subheader("予測期間のサマリー")
            
            # 予測日数
            forecast_period_days = forecast_days
            
            # 合計値
            total_quantity = np.sum(forecast_quantity)
            total_amount = np.sum(forecast_amount)
            
            # サマリーを表示
            col_sum1, col_sum2, col_sum3, col_sum4 = st.columns(4)
            with col_sum1:
                st.metric("予測期間", f"{forecast_period_days}日")
            with col_sum2:
                st.metric("合計売上数量", f"{total_quantity:,.1f}")
            with col_sum3:
                st.metric("合計税抜売上金額", f"{total_amount:,.0f}円")
            with col_sum4:
                st.metric("平均1日あたり", f"{total_amount/forecast_period_days:,.0f}円")
            
            # 予測結果のテーブル表示
            st.markdown("---")
            st.subheader("予測結果（日別）")
            
            # 配列の長さを確認し、一致させる
            min_length = min(len(forecast_dates_quantity), len(forecast_quantity), len(forecast_amount))
            forecast_df = pd.DataFrame({
                '日付': forecast_dates_quantity[:min_length],
                '売上数量（予測）': forecast_quantity[:min_length].round(2),
                '税抜売上金額（予測）': forecast_amount[:min_length].round(0)
            })
            # 日付をyyyy/mm/dd形式に変換
            forecast_df['日付'] = forecast_df['日付'].dt.strftime('%Y/%m/%d')
            # 売上数量を小数点2位まで表示
            forecast_df['売上数量（予測）'] = forecast_df['売上数量（予測）'].apply(lambda x: f"{x:,.2f}")
            # 税抜売上金額をフォーマット
            forecast_df['税抜売上金額（予測）'] = forecast_df['税抜売上金額（予測）'].apply(lambda x: f"{x:,.0f}")
            
            # カスタムCSSで右寄せを適用
            st.markdown("""
            <style>
            .dataframe td:nth-child(2),
            .dataframe td:nth-child(3) {
                text-align: right !important;
            }
            </style>
            """, unsafe_allow_html=True)
            
            st.dataframe(
                forecast_df,
                use_container_width=True,
                hide_index=True
            )
            
            # 統計サマリー（平均値など）
            col3, col4, col5 = st.columns(3)
            with col3:
                st.metric("予測期間の平均売上数量", f"{forecast_quantity.mean():.1f}")
            with col4:
                st.metric("予測期間の平均税抜売上金額", f"{np.mean(forecast_amount):,.0f}円")
            with col5:
                st.metric("最大1日あたり金額", f"{np.max(forecast_amount):,.0f}円")
            
            # CSVダウンロード
            st.markdown("---")
            csv = forecast_df.to_csv(index=False, encoding='utf-8-sig')
            st.download_button(
                label="予測結果をCSVでダウンロード",
                data=csv,
                file_name=f"販売予測_{selected_model}_{start_date.strftime('%Y%m%d')}_{end_date.strftime('%Y%m%d')}.csv",
                mime="text/csv"
            )
        
        else:
            # 実績データのみ表示（予測に使用するデータを表示）
            st.subheader("売上数量の推移")
            fig_quantity = go.Figure()
            fig_quantity.add_trace(go.Scatter(
                x=daily_data_for_forecast['日付'],
                y=daily_data_for_forecast['売上数量'],
                mode='lines+markers',
                name='実績',
                line=dict(color='#1f77b4', width=2),
                marker=dict(size=6)
            ))
            fig_quantity.update_layout(
                title='売上数量の推移',
                xaxis_title='日付',
                yaxis_title='売上数量',
                hovermode='x unified',
                height=400,
                template='plotly_white'
            )
            st.plotly_chart(fig_quantity, use_container_width=True)
            
            st.info("👆 サイドバーで予測モデルと期間を設定し、「予測を実行」ボタンをクリックしてください。")
        
        # データサマリー
        with st.expander("データサマリーを表示"):
            st.dataframe(daily_data, use_container_width=True)

else:
    st.info("👈 サイドバーからCSVファイルをアップロードしてください。")
    
    if uploaded_file is None:
        st.markdown("""
        ### 使い方
        
        1. サイドバーからCSVファイルをアップロード
        2. 予測モデルを選択
        3. 予測期間を設定
        4. 「予測を実行」ボタンをクリック
        
        ### 対応している予測モデル
        
        - **Prophet**: Facebookが開発した時系列予測モデル。季節性やトレンドを自動検出します。
        - **ARIMA**: 自己回帰移動平均モデル。統計的に堅牢な予測が可能です。
        - **Linear Regression**: 線形回帰モデル。シンプルで理解しやすい予測が可能です。
        """)

