import pandas as pd
import numpy as np
import json
import os

# ==========================================
# 設定項目
# ==========================================
BASE_DIR = '/Users/user/Downloads/short_ar_usage_dataset' # 点群データのベースディレクトリ
INPUT_FILE = '/Users/user/Downloads/short_ar_usage_dataset_proj_lora_results.xlsx'               # 入力ファイル名
OUTPUT_FILE = 'proj_lora_point_cloud_dashboard_short.html'     # 出力ファイル名
LIMIT = 50                                     # 処理するデータ数の上限（Noneで全件）

# ==========================================
# メイン処理
# ==========================================

def load_point_cloud_data(npy_path):
    """
    .npyファイルを読み込み、JSONシリアライズ可能な形式(リスト)で返す
    データ量を減らすため、座標は小数点以下3桁、色は整数に丸める
    """
    try:
        # [x, y, z, r, g, b, class_id, subclass_id]
        data = np.load(npy_path)
        
        # 間引き（オプション）: 点が多すぎて重い場合はここで data[::2] などしてください
        # data = data[::2]

        xyz = data[:, :3]
        rgb = data[:, 3:6]
        
        # RGBの正規化チェック (0-1なら0-255に変換)
        if rgb.max() <= 1.0:
            rgb = (rgb * 255).astype(int)
        else:
            rgb = rgb.astype(int)
            
        # 色文字列の作成 'rgb(r,g,b)'
        color_strs = [f'rgb({r},{g},{b})' for r, g, b in rgb]
        
        return {
            "x": np.round(xyz[:, 0], 4).tolist(),
            "y": np.round(xyz[:, 1], 4).tolist(),
            "z": np.round(xyz[:, 2], 4).tolist(),
            "color": color_strs
        }
    except Exception as e:
        print(f"  [Error] Could not load {npy_path}: {e}")
        return None

def generate_dashboard():
    print("Loading Excel file...")
    if INPUT_FILE.endswith('.csv'):
        df = pd.read_csv(INPUT_FILE)
    else:
        df = pd.read_excel(INPUT_FILE)

    if LIMIT:
        df = df.head(LIMIT)
        print(f"Processing first {LIMIT} rows...")

    # データをJSON構造に変換
    dataset = []
    
    for index, row in df.iterrows():
        print(f"Processing {index+1}/{len(df)}: ID {row.get('ID', 'N/A')}")
        
        rel_path = row['point_cloud']
        full_path = os.path.join(BASE_DIR, rel_path)
        
        pc_data = load_point_cloud_data(full_path)
        
        if pc_data:
            # -------------------------------------------------
            # 修正箇所: ASSISTANT: 以降のテキストのみを抽出
            # -------------------------------------------------
            raw_pred = str(row.get('prediction', ''))
            if "ASSISTANT:" in raw_pred:
                # "ASSISTANT:" で分割し、最後の部分を取得して空白除去
                clean_pred = raw_pred.split("ASSISTANT:")[-1].strip()
            else:
                # 見つからない場合はそのまま
                clean_pred = raw_pred
            # -------------------------------------------------

            entry = {
                "id": str(row.get('ID', index)),
                "path": rel_path,
                "question": str(row.get('question', '')),
                "ground_truth": str(row.get('ground_truth', '')),
                "generated": clean_pred,  # 抽出したテキストを使用
                "pc": pc_data
            }
            dataset.append(entry)

    # JSON文字列化（HTMLに埋め込むため）
    json_data = json.dumps(dataset)

    print("Generating HTML...")
    
    html_content = f"""
<!DOCTYPE html>
<html lang="en">
<head>
    <meta charset="UTF-8">
    <meta name="viewport" content="width=device-width, initial-scale=1.0">
    <title>Point Cloud VQA Dashboard</title>
    <script src="https://cdn.plot.ly/plotly-2.27.0.min.js"></script>
    <style>
        body {{ margin: 0; padding: 0; font-family: -apple-system, BlinkMacSystemFont, "Segoe UI", Roboto, Helvetica, Arial, sans-serif; height: 100vh; display: flex; overflow: hidden; }}
        
        /* 左サイドバー（リスト） */
        #sidebar {{
            width: 300px;
            background: #f0f2f5;
            border-right: 1px solid #ddd;
            overflow-y: auto;
            display: flex;
            flex-direction: column;
        }}
        .list-header {{ padding: 15px; background: #fff; border-bottom: 1px solid #ddd; font-weight: bold; }}
        .list-item {{
            padding: 12px 15px;
            border-bottom: 1px solid #e0e0e0;
            cursor: pointer;
            transition: background 0.2s;
            font-size: 0.9rem;
        }}
        .list-item:hover {{ background: #e6eef5; }}
        .list-item.active {{ background: #007bff; color: white; }}
        .item-id {{ font-weight: bold; display: block; margin-bottom: 4px; }}
        .item-preview {{ font-size: 0.8rem; opacity: 0.8; white-space: nowrap; overflow: hidden; text-overflow: ellipsis; }}

        /* メインエリア */
        #main {{ flex: 1; display: flex; flex-direction: column; height: 100vh; }}
        
        /* 3Dビューア */
        #viewer-container {{ flex: 3; position: relative; border-bottom: 1px solid #ddd; min-height: 400px; }}
        #plotly-div {{ width: 100%; height: 100%; }}
        
        /* テキスト情報エリア */
        #info-panel {{ 
            flex: 2; 
            padding: 20px; 
            overflow-y: auto; 
            background: #fff; 
            display: flex; 
            gap: 20px;
        }}
        .info-col {{ flex: 1; min-width: 200px; }}
        .info-box {{ background: #f9f9f9; padding: 15px; border-radius: 6px; border: 1px solid #eee; height: 100%; box-sizing: border-box; }}
        .info-label {{ font-weight: bold; color: #555; margin-bottom: 8px; font-size: 0.85rem; text-transform: uppercase; letter-spacing: 0.5px; }}
        .info-text {{ white-space: pre-wrap; font-size: 0.95rem; line-height: 1.5; color: #333; }}
        
        .gt-box {{ border-left: 4px solid #28a745; }}
        .gen-box {{ border-left: 4px solid #007bff; }}
        
        /* ローディング表示 */
        #loading {{ position: absolute; top: 50%; left: 50%; transform: translate(-50%, -50%); font-size: 1.2rem; color: #666; display: none; }}
    </style>
</head>
<body>

<div id="sidebar">
    <div class="list-header">Data Samples ({len(dataset)})</div>
    <div id="list-container"></div>
</div>

<div id="main">
    <div id="viewer-container">
        <div id="plotly-div"></div>
        <div id="loading">Loading...</div>
    </div>
    <div id="info-panel">
        <div class="info-col">
            <div class="info-box">
                <div class="info-label">Question</div>
                <div class="info-text" id="text-question"></div>
            </div>
        </div>
        <div class="info-col">
            <div class="info-box gt-box">
                <div class="info-label">Ground Truth</div>
                <div class="info-text" id="text-gt"></div>
            </div>
        </div>
        <div class="info-col">
            <div class="info-box gen-box">
                <div class="info-label">Generated Output</div>
                <div class="info-text" id="text-gen"></div>
            </div>
        </div>
    </div>
</div>

<script>
    // Pythonから埋め込まれたデータ
    const allData = {json_data};
    let currentIndex = -1;

    // 初期化
    function init() {{
        const listContainer = document.getElementById('list-container');
        
        allData.forEach((item, idx) => {{
            const div = document.createElement('div');
            div.className = 'list-item';
            div.innerHTML = `<span class="item-id">ID: ${{item.id}}</span><span class="item-preview">${{item.question}}</span>`;
            div.onclick = () => loadItem(idx);
            div.id = 'item-' + idx;
            listContainer.appendChild(div);
        }});

        if (allData.length > 0) {{
            loadItem(0);
        }}
    }}

    // アイテムのロードと表示
    function loadItem(index) {{
        if (currentIndex === index) return;
        
        // リストのハイライト更新
        if (currentIndex !== -1) {{
            document.getElementById('item-' + currentIndex).classList.remove('active');
        }}
        currentIndex = index;
        document.getElementById('item-' + currentIndex).classList.add('active');

        const item = allData[index];

        // テキスト更新
        document.getElementById('text-question').innerText = item.question;
        document.getElementById('text-gt').innerText = item.ground_truth;
        document.getElementById('text-gen').innerText = item.generated;

        // 3Dプロット更新
        renderPlot(item);
    }}

    function renderPlot(item) {{
        const trace = {{
            x: item.pc.x,
            y: item.pc.y,
            z: item.pc.z,
            mode: 'markers',
            marker: {{
                size: 2,
                color: item.pc.color,
                opacity: 0.8
            }},
            type: 'scatter3d'
        }};

        const layout = {{
            margin: {{l: 0, r: 0, b: 0, t: 0}},
            scene: {{
                aspectmode: 'data',
                xaxis: {{ title: 'X' }},
                yaxis: {{ title: 'Y' }},
                zaxis: {{ title: 'Z' }},
                bgcolor: '#fafafa'
            }}
        }};

        const config = {{ responsive: true }};

        // 新規プロット作成（Reactの方が効率的ですが、シーンのリセット等のためにnewPlotを使用）
        Plotly.newPlot('plotly-div', [trace], layout, config);
    }}

    // 実行開始
    init();

</script>
</body>
</html>
    """

    with open(OUTPUT_FILE, 'w', encoding='utf-8') as f:
        f.write(html_content)
    
    print(f"Done! Created {OUTPUT_FILE}")
    print("Open this file in your browser to view the interactive dashboard.")

if __name__ == "__main__":
    generate_dashboard()