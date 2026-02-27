import argparse
import Levenshtein
import json

def calculate_anls(prediction: str, ground_truths: list[str], threshold: float = 0.5) -> float:
    """
    Calcola l'ANLS tra la predizione del modello e le risposte corrette.
    """
    prediction = str(prediction).strip().lower()
    max_score = 0.0
    
    for gt in ground_truths:
        gt = str(gt).strip().lower()
        
        if len(prediction) == 0 and len(gt) == 0:
            score = 1.0
        elif len(prediction) == 0 or len(gt) == 0:
            score = 0.0
        else:
            dist = Levenshtein.distance(prediction, gt)
            max_len = max(len(prediction), len(gt))
            normalized_dist = dist / max_len
            
            # Se l'errore è sotto la soglia (0.5), assegna il punteggio parziale
            if normalized_dist < threshold:
                score = 1.0 - normalized_dist
            else:
                score = 0.0
                
        if score > max_score:
            max_score = score
            
    return max_score

def compute_score_docvqa(predictions_file: str) -> float:
    with open(predictions_file, 'r') as f:
        data = json.load(f)
    
    anls_scores = []
    
    for element in data:
        reference_list = element['ground_truths']
        candidate = element['prediction']
        
        score = calculate_anls(candidate, reference_list)
        anls_scores.append(score)
    
    return sum(anls_scores) / len(anls_scores) if anls_scores else 0.0