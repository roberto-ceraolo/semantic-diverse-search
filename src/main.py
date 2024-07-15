import logging
from pair_finder import SentencePairFinder
from evaluator import Evaluator
from utils import Config, setup_logging
from datetime import datetime


def main():
    config = Config('config.yaml')
    log_filepath = setup_logging(config)
    experiment_id = log_filepath.split('/')[-1][:-4]

    start_time = datetime.now()
    
    finder = SentencePairFinder(config)
    evaluator = Evaluator(config)

    en_diverse_pairs, de_diverse_pairs, cross_lingual_pairs, en_stats, de_stats = finder.run(experiment_id)

    training_stats = {
        "en": en_stats,
        "de": de_stats
    }

    evaluator.log_training_stats(training_stats)
    
    eval_results = {
        "en": evaluator.evaluate_pairs(en_diverse_pairs, "en"),
        "de": evaluator.evaluate_pairs(de_diverse_pairs, "de"),
        "cross_lingual": evaluator.evaluate_pairs(cross_lingual_pairs, "other")
    }

    evaluator.log_results(eval_results)
    
    if config.generate_csv:
        logging.info(f"CSV files have been generated in the '{config.csv_output_dir}' directory.")
    
    logging.info("Evaluation complete. Results have been logged to file.")
    
    
    end_time = datetime.now()
    duration_seconds = (end_time - start_time).total_seconds()
    duration_minutes = duration_seconds / 60
    logging.info(f"Total run duration: {duration_minutes:.2f} minutes")



if __name__ == "__main__":
    main()