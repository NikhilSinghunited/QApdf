import unittest
from unittest.mock import patch, MagicMock
import streamlit as st
from io import BytesIO
from app import get_pdf_text, get_text_chunks, get_vector_store, user_input, get_conversational_chain

class TestAppFunctions(unittest.TestCase):

    @patch("app.PdfReader")
    def test_get_pdf_text(self, MockPdfReader):
        # Mocking the PDF reading functionality
        mock_pdf_file = MagicMock()
        mock_pdf_file.name = "test.pdf"
        mock_pdf_file.read.return_value = b"Some text content"

        mock_pdf_reader = MagicMock()
        mock_pdf_reader.pages = [MagicMock(extract_text=MagicMock(return_value="Some text content"))]
        MockPdfReader.return_value = mock_pdf_reader

        # Test with a single PDF
        pdf_docs = [mock_pdf_file]
        result = get_pdf_text(pdf_docs)

        self.assertIn("Some text content", result)
        MockPdfReader.assert_called_once_with(mock_pdf_file)

    def test_get_text_chunks(self):
        # Test text splitting
        raw_text = "This is a sample text that should be split into chunks based on the size limit."
        result = get_text_chunks(raw_text)
        
        # Assuming the chunk size is 10 characters, the result should be a list of smaller strings
        self.assertTrue(len(result) > 1)  # Since the text is longer than the chunk size

    @patch("app.FAISS.from_texts")
    def test_get_vector_store(self, MockFAISS):
        # Mocking the FAISS vector store creation
        mock_text_chunks = ["chunk1", "chunk2", "chunk3"]
        embeddings = MagicMock()
        
        mock_vector_store = MagicMock()
        MockFAISS.from_texts.return_value = mock_vector_store
        
        get_vector_store(mock_text_chunks)

        MockFAISS.from_texts.assert_called_once_with(mock_text_chunks, embedding=embeddings)
        mock_vector_store.save_local.assert_called_once_with("faiss_index")

    @patch("app.FAISS.load_local")
    @patch("app.GoogleGenerativeAIEmbeddings")
    @patch("app.get_conversational_chain")
    def test_user_input(self, MockConversationalChain, MockEmbeddings, MockFAISS):
        # Mocking the FAISS load, embedding and chain behavior
        mock_vector_store = MagicMock()
        MockFAISS.load_local.return_value = mock_vector_store
        
        mock_embeddings = MagicMock()
        MockEmbeddings.return_value = mock_embeddings
        
        mock_chain = MagicMock()
        MockConversationalChain.return_value = mock_chain
        mock_chain.return_value = {"output_text": "This is the answer to the query."}

        user_question = "What is the experience of candidate X?"
        user_input(user_question)

        # Ensure the FAISS load_local function was called
        MockFAISS.load_local.assert_called_once_with("faiss_index", mock_embeddings, allow_dangerous_deserialization=True)

        # Ensure the conversational chain was called
        MockConversationalChain.assert_called_once()

        # Ensure the response is written to the Streamlit UI
        st.write.assert_called_once_with("Reply: ", "This is the answer to the query.")

    @patch("app.get_conversational_chain")
    @patch("app.FAISS.load_local")
    def test_empty_user_input(self, MockFAISS, MockConversationalChain):
        # Testing the case when user input is empty
        user_input("")

        # Ensure that no interaction happens when input is empty
        MockFAISS.load_local.assert_not_called()
        MockConversationalChain.assert_not_called()

    @patch("app.FAISS.load_local")
    def test_invalid_pdf(self, MockFAISS):
        # Test with invalid or empty PDF text
        invalid_pdf_docs = []
        raw_text = get_pdf_text(invalid_pdf_docs)

        self.assertEqual(raw_text, "")

    def test_invalid_question(self):
        # Test invalid question (empty)
        with self.assertRaises(ValueError):
            user_input("")

    @patch("app.st.spinner")
    @patch("app.st.success")
    def test_process_pdf_button(self, MockSuccess, MockSpinner):
        # Test the processing of PDF in the sidebar
        pdf_file = MagicMock()
        pdf_file.name = "test.pdf"
        pdf_docs = [pdf_file]
        
        # Simulate the button click and file upload
        with patch("app.get_pdf_text") as MockGetPdfText, patch("app.get_text_chunks") as MockGetTextChunks:
            MockGetPdfText.return_value = "Sample text from PDF"
            MockGetTextChunks.return_value = ["chunk1", "chunk2"]
            
            # Mock the process in the sidebar
            with patch("app.get_vector_store") as MockGetVectorStore:
                st.button("Submit & Process")
                MockGetVectorStore.assert_called_once_with(["chunk1", "chunk2"])
                MockSuccess.assert_called_once_with("Files processed successfully! You can now ask questions.")
                MockSpinner.assert_called_once()

if __name__ == "__main__":
    unittest.main()
