import unittest
import sys
from unittest.mock import patch, MagicMock

# Assuming the test file is in the same directory as main.py
# If not, you might need to adjust the path.
sys.path.append('../..')

from MinLlama.Running.main import setup_args_for_option, main


class TestMain(unittest.TestCase):

    def test_setup_args_for_generate_option(self):
        """
        Tests argument setup for the 'generate' option.
        """
        option = "generate"
        args = setup_args_for_option(option)
        self.assertEqual(args.option, "generate")
        self.assertIsNone(args.train, "Train data should be None for generate option")
        self.assertEqual(args.pretrained_model_path, "../SanityCheck/stories42M.pt")

    def test_setup_args_for_prompt_option(self):
        """
        Tests argument setup for the 'prompt' option.
        """
        option = "prompt"
        args = setup_args_for_option(option)
        self.assertEqual(args.option, "prompt")
        self.assertEqual(args.train, "../Data/sst-train.txt")
        self.assertEqual(args.dev, "../Data/sst-dev.txt")
        self.assertEqual(args.test, "../Data/sst-test.txt")
        self.assertEqual(args.label_names, "../Data/sst-label-mapping.json")
        self.assertEqual(args.dev_out, "cfimdb-dev-prompting-output.txt")
        self.assertEqual(args.test_out, "cfimdb-test-prompting-output.txt")

    def test_setup_args_for_finetune_option(self):
        """
        Tests argument setup for the 'finetune' option.
        """
        option = "finetune"
        args = setup_args_for_option(option)
        self.assertEqual(args.option, "finetune")
        self.assertEqual(args.train, "../Data/sst-train.txt")
        self.assertEqual(args.dev, "../Data/sst-dev.txt")
        self.assertEqual(args.test, "../Data/sst-test.txt")
        self.assertEqual(args.label_names, "../Data/sst-label-mapping.json")
        self.assertEqual(args.dev_out, "sst-dev-finetuning-output.txt")
        self.assertEqual(args.test_out, "sst-test-finetuning-output.txt")

    @patch('MinLlama.Running.main.seed_everything')
    @patch('MinLlama.Running.main.generate_sentence')
    def test_main_generate(self, mock_generate_sentence, mock_seed_everything):
        """
        Tests that main calls generate_sentence for the 'generate' option.
        """
        main("generate")
        mock_seed_everything.assert_called_once()
        self.assertEqual(mock_generate_sentence.call_count, 2)

    @patch('MinLlama.Running.main.seed_everything')
    @patch('MinLlama.Running.main.test_with_prompting')
    def test_main_prompt(self, mock_test_with_prompting, mock_seed_everything):
        """
        Tests that main calls test_with_prompting for the 'prompt' option.
        """
        main("prompt")
        mock_seed_everything.assert_called_once()
        mock_test_with_prompting.assert_called_once()

    @patch('MinLlama.Running.main.seed_everything')
    @patch('MinLlama.Running.main.train')
    @patch('MinLlama.Running.main.test')
    def test_main_finetune(self, mock_test, mock_train, mock_seed_everything):
        """
        Tests that main calls train and test for the 'finetune' option.
        """
        main("finetune")
        mock_seed_everything.assert_called_once()
        mock_train.assert_called_once()
        mock_test.assert_called_once()

    def test_main_invalid_option(self):
        """
        Tests that main raises a ValueError for an invalid option.
        """
        with self.assertRaises(ValueError):
            main("invalid_option")


if __name__ == "__main__":
    # This allows running tests directly from the command line
    # You'll need to be in the 'Running' directory and run: python test_main.py
    # Or use a test runner like pytest.
    # To run with unittest runner: python -m unittest test_main.py
    unittest.main()