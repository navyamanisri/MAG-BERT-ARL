"""
MAG-BERT-ARL Pipeline Integration Test

This unit test validates that the interview pipeline and associated service skeletons
integrate smoothly and execute the full end-to-end flow correctly.
"""

import unittest
import os
from pathlib import Path
from backend.pipelines.interview_pipeline import process_interview

class TestInterviewPipeline(unittest.TestCase):
    def setUp(self):
        """Creates a temporary mock video file to satisfy path existence checks."""
        self.test_video_path = "mock_test_interview.mp4"
        with open(self.test_video_path, "w") as f:
            f.write("mock video content")

    def tearDown(self):
        """Cleans up all temporary mock video files and simulated service output paths."""
        if os.path.exists(self.test_video_path):
            os.remove(self.test_video_path)
        
        # Clean up simulated wav file
        mock_wav = self.test_video_path.replace(".mp4", ".wav")
        if os.path.exists(mock_wav):
            os.remove(mock_wav)
            
        # Clean up simulated visual features file
        mock_visual = self.test_video_path.replace(".mp4", "_visual_features.npy")
        if os.path.exists(mock_visual):
            os.remove(mock_visual)
            
        # Clean up simulated acoustic features file
        mock_acoustic = self.test_video_path.replace(".mp4", "_acoustic_features.csv")
        if os.path.exists(mock_acoustic):
            os.remove(mock_acoustic)

    def test_pipeline_execution(self):
        """Executes the pipeline on the mock video and asserts correct outputs and schemas."""
        print("\n" + "=" * 50)
        print("RUNNING PIPELINE INTEGRATION TEST")
        print("=" * 50)
        
        # Run the full process
        result = process_interview(self.test_video_path)
        
        # 1. Assert successful pipeline completion status
        self.assertEqual(result["status"], "success")
        self.assertEqual(result["video_path"], self.test_video_path)
        
        # 2. Validate metadata values
        self.assertEqual(result["metadata"]["file_name"], "mock_test_interview.mp4")
        self.assertEqual(result["metadata"]["resolution"], "1920x1080")
        
        # 3. Assert presence of all three modalities in the structured response
        self.assertIn("modalities", result)
        self.assertIn("visual", result["modalities"])
        self.assertIn("acoustic", result["modalities"])
        self.assertIn("verbal", result["modalities"])
        
        # 4. Verify visual features structure
        visual = result["modalities"]["visual"]
        self.assertEqual(visual["status"], "extracted")
        self.assertTrue(visual["features_path"].endswith("_visual_features.npy"))
        
        # 5. Verify acoustic features structure
        acoustic = result["modalities"]["acoustic"]
        self.assertEqual(acoustic["status"], "extracted")
        self.assertTrue(acoustic["features_path"].endswith("_acoustic_features.csv"))
        
        # 6. Verify verbal transcription and tokenization structure
        verbal = result["modalities"]["verbal"]
        self.assertEqual(verbal["status"], "tokenized")
        self.assertTrue(len(verbal["transcription"]) > 0)
        self.assertEqual(verbal["token_ids"], [101, 7592, 1010, 102])
        
        print("\n" + "=" * 50)
        print("PIPELINE INTEGRATION TEST PASSED SUCCESSFULLY")
        print("=" * 50)

if __name__ == "__main__":
    unittest.main()
