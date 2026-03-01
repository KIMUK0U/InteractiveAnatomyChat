"""
Rule-based Prompt Generator - Simple, Fast, Reliable (CORRECTED VERSION)
シンプル、高速、信頼性の高いルールベース実装

IMPORTANT FIX:
- Removed automatic <point> token insertion
- Token sequence will be added by build_dataset script
- This ensures proper <point_start><point_patch>...<point_end> format
"""

from typing import Tuple, Optional
import random


COLORIZATION_EFFECTS = {
    "hover": {
        # 黄色く光るのではなく、純粋に「明るく」する表現へ修正
        "verb": "brightened", 
        "adjective": "brightened with increased luminosity", 
        "noun": "brightness enhancement"
    },
    "contrast": {
        # 明るい色ではなく、「補色」であることを明記
        "verb": "colored",
        "adjective": "shown in complementary colors",
        "noun": "complementary color highlighting"
    },
    "fixed_blue": {
        "verb": "colored",
        "adjective": "colored in blue",
        "noun": "blue coloring"
    },
    "fixed_red": {
        "verb": "colored",
        "adjective": "colored in red",
        "noun": "red coloring"
    },
    "fixed_black": {
        "verb": "darkened",
        "adjective": "darkened",
        "noun": "dark effect"
    },
}


class RuleBasedPromptGenerator:
    """
    ルールベースのプロンプト生成
    
    利点:
    - 100%制約を守る（空間情報を絶対に含まない）
    - 高速（API呼び出し不要）
    - コスト0
    - デバッグ簡単
    - 十分な多様性（6パターン × 色バリエーション × データ多様性）
    
    CORRECTED VERSION NOTES:
    - Does NOT add <point> token automatically
    - Point token sequence should be added by the caller
    - This allows proper <point_start><point_patch>...<point_end> format
    """
    
    def __init__(self, seed: int = 42):
        random.seed(seed)
    
    def generate(
        self,
        colorization_type: str,
        primary_region,
        secondary_region: Optional[any],
        all_candidates: list,
        distance_mm: float,
        variation_index: int = 0
    ) -> Tuple[str, str]:
        """
        質問と答えのペアを生成
        
        質問: 視覚的事実のみ（色、距離、48クラスリスト）
        答え: 空間推論を含む
        
        NOTE: Point token sequence is NOT added here.
        The caller should prepend the proper token sequence:
        <point_start><point_patch>...(513 times)...<point_end>
        
        Returns:
            (question, answer) tuple
            - question: WITHOUT point tokens (caller adds them)
            - answer: Complete answer text
        """
        
        color_info = COLORIZATION_EFFECTS.get(
            colorization_type, 
            COLORIZATION_EFFECTS["contrast"]
        )
        
        # 全候補のリスト文字列と数
        all_candidates_str = ", ".join(all_candidates)
        num_candidates = len(all_candidates)
        
        # トップ候補
        top_candidates = [primary_region.anatomical_subclass_name]
        if secondary_region:
            top_candidates.append(secondary_region.anatomical_subclass_name)
        top_candidates_str = ", ".join(top_candidates)
        
        # 空間記述（答えのみ）
        spatial_desc = self._extract_spatial_description(
            primary_region.anatomical_subclass_name
        )
        
        # ===== QUESTION TEMPLATES =====
        # 6パターンで十分な多様性
        
        question_templates = [
            # Pattern 0: フォーマル・詳細
            (
                f"This dental CBCT model contains {num_candidates} anatomical structures: [{all_candidates_str}]. "
                f"In the point cloud visualization, one region has been {color_info['verb']} "
                f"({color_info['adjective']}) to indicate where the user's hand (tracked in AR) "
                f"is pointing, approximately {distance_mm:.1f}mm from the surface. "
                f"Based on the spatial distribution and color pattern visible in the point cloud, "
                f"which specific anatomical structure is being indicated?"
            ),
            
            # Pattern 1: カジュアル・短め
            (
                f"The model has {num_candidates} labeled regions: [{all_candidates_str}]. "
                f"Looking at the point cloud, one area is {color_info['adjective']} "
                f"where the user's hand is pointing (about {distance_mm:.1f}mm away). "
                f"Which anatomical structure is this?"
            ),
            
            # Pattern 2: 技術的
            (
                f"This AR-enhanced dental CBCT visualization displays {num_candidates} anatomical structures: "
                f"[{all_candidates_str}]. The point cloud shows {color_info['noun']} "
                f"at the region of interest, with hand tracking at {distance_mm:.1f}mm proximity. "
                f"Identify the corresponding anatomical structure."
            ),
            
            # Pattern 3: 質問先行
            (
                f"Which anatomical structure from these {num_candidates} dental regions [{all_candidates_str}] "
                f"corresponds to the area that has been {color_info['verb']} in the point cloud "
                f"(user's hand: {distance_mm:.1f}mm away)?"
            ),
            
            # Pattern 4: 観察的
            (
                f"Examining this dental point cloud with {num_candidates} possible structures [{all_candidates_str}]: "
                f"One region shows {color_info['noun']}, indicating the user's focus point "
                f"at {distance_mm:.1f}mm hand distance. What structure is indicated?"
            ),
            
            # Pattern 5: 直接的・簡潔
            (
                f"Point cloud with {num_candidates} anatomical regions: [{all_candidates_str}]. "
                f"The {color_info['verb']} area (hand: {distance_mm:.1f}mm) indicates which structure?"
            ),
        ]
        
        # ===== ANSWER TEMPLATES =====
        
        answer_templates = [
            # Pattern 0: 標準的なChain-of-Thought
            (
                f"The main anatomical category here is {primary_region.anatomical_class_name}. "
                f"Analyzing the spatial location of the {color_info['verb']} region in the point cloud, "
                f"it appears to be {spatial_desc}. "
                f"From all {num_candidates} possible anatomical regions, the candidates are: {top_candidates_str}. "
                f"Therefore, the indicated region corresponds to {primary_region.anatomical_subclass_name}."
            ),
            
            # Pattern 1: 詳細推論
            (
                f"Looking at the {color_info['verb']} area in the point cloud: "
                f"The main anatomical category is {primary_region.anatomical_class_name}. "
                f"Based on the spatial distribution, this region is located {spatial_desc}. "
                f"Considering all {num_candidates} structures, the top candidates are {top_candidates_str}, "
                f"with {primary_region.anatomical_subclass_name} being the match."
            ),
            
            # Pattern 2: 空間優先
            (
                f"The {color_info['verb']} region appears to be {spatial_desc}, "
                f"which places it in the {primary_region.anatomical_class_name} category. "
                f"From the {num_candidates} possible anatomical structures, this corresponds to "
                f"{primary_region.anatomical_subclass_name}."
            ),
            
            # Pattern 3: 消去法的推論
            (
                f"Starting from {num_candidates} possible structures, the spatial location of the {color_info['verb']} area "
                f"({spatial_desc}) narrows it to the {primary_region.anatomical_class_name} category. "
                f"The specific candidates are {top_candidates_str}. "
                f"Answer: {primary_region.anatomical_subclass_name}."
            ),
            
            # Pattern 4: 簡潔
            (
                f"The {color_info['verb']} region indicates {primary_region.anatomical_class_name}, "
                f"specifically {spatial_desc}. "
                f"Among {num_candidates} structures, this is {primary_region.anatomical_subclass_name}."
            ),
            
            # Pattern 5: ステップバイステップ
            (
                f"Step 1: Category identification - {primary_region.anatomical_class_name}. "
                f"Step 2: Spatial analysis - {spatial_desc}. "
                f"Step 3: From {num_candidates} candidates, narrow to: {top_candidates_str}. "
                f"Conclusion: {primary_region.anatomical_subclass_name}."
            ),
        ]
        
        # Select by variation index
        question = question_templates[variation_index % len(question_templates)]
        answer = answer_templates[variation_index % len(answer_templates)]
        
        # IMPORTANT: Do NOT add <point> token here
        # The caller will add the proper token sequence
        
        return question, answer
    
    def _extract_spatial_description(self, anatomical_name: str) -> str:
        """解剖学的名前から空間記述を抽出"""
        name_lower = anatomical_name.lower()
        parts = []
        
        # Side
        if "right" in name_lower:
            parts.append("on the right side")
        elif "left" in name_lower:
            parts.append("on the left side")
        
        # Anterior/Posterior
        if "posterior" in name_lower:
            parts.append("in the posterior (back) region")
        elif "anterior" in name_lower:
            parts.append("in the anterior (front) region")
        
        # Superior/Inferior
        if "superior" in name_lower:
            parts.append("in the superior (upper) portion")
        elif "inferior" in name_lower:
            parts.append("in the inferior (lower) portion")
        
        # Tooth type
        if "#8" in name_lower or "#7" in name_lower or "#6" in name_lower:
            parts.append("in the molar region (back teeth)")
        elif "#5" in name_lower or "#4" in name_lower:
            parts.append("in the premolar region")
        elif "#3" in name_lower:
            parts.append("in the canine region")
        elif "#2" in name_lower or "#1" in name_lower:
            parts.append("in the incisor region (front teeth)")
        
        return ", ".join(parts) if parts else "in the central region"


# Usage example
"""
from build_dataset import SUBCLASS_NAME_MAP, format_point_token_sequence

generator = RuleBasedPromptGenerator(seed=42)

# Generate question and answer (WITHOUT point tokens)
question, answer = generator.generate(
    colorization_type="blue",
    primary_region=region_obj,
    secondary_region=None,
    all_candidates=list(SUBCLASS_NAME_MAP.values()),
    distance_mm=2.5,
    variation_index=0  # 0-5 for different styles
)

# Add proper point token sequence
point_tokens = format_point_token_sequence()  # <point_start><point_patch>...<point_end>
question_with_tokens = f"{point_tokens}\n{question}"

print("Question:", question_with_tokens)
print("Answer:", answer)
"""