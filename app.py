import streamlit as st
import pandas as pd
import fitz  # PyMuPDF is imported as fitz
import re
import json
import io
import time
from typing import TypedDict, List, Dict, Any, Annotated
from dataclasses import dataclass
from enum import Enum
import asyncio
import threading
from concurrent.futures import ThreadPoolExecutor
import openpyxl
from openpyxl.utils.dataframe import dataframe_to_rows
from openpyxl.styles import Font, PatternFill, Alignment
import spacy
from sentence_transformers import SentenceTransformer
import torch
import google.generativeai as genai
from langgraph.graph import StateGraph, START, END
from langgraph.graph.message import add_messages
from langgraph.checkpoint.memory import MemorySaver
import nltk
from nltk.tokenize import sent_tokenize, word_tokenize
import logging
import uuid

# Configure logging
logging.basicConfig(level=logging.INFO)
logger = logging.getLogger(__name__)

# Download required NLTK data
try:
    nltk.download('punkt', quiet=True)
    nltk.download('punkt_tab', quiet=True)
except:
    pass

# Initialize session state
if 'processing_complete' not in st.session_state:
    st.session_state.processing_complete = False
if 'results_df' not in st.session_state:
    st.session_state.results_df = None
if 'progress_messages' not in st.session_state:
    st.session_state.progress_messages = []
if 'document_text_content' not in st.session_state:
    st.session_state.document_text_content = ""

class IGComponentType(Enum):
    """Institutional Grammar component types based on IG 2.0 specification"""
    ATTRIBUTE = "A"  # Actor
    DEONTIC = "D"   # Deontic operator (may, shall, must, etc.)
    AIM = "I"       # Action/Aim
    OBJECT = "B"    # Direct object
    ACTIVATION_CONDITION = "Cac"  # Context activation condition
    EXECUTION_CONSTRAINT = "Cex"  # Context execution constraint
    OR_ELSE = "O"   # Consequence

class StatementType(Enum):
    """Types of institutional statements"""
    REGULATIVE = "regulative"
    CONSTITUTIVE = "constitutive"
    HYBRID = "hybrid"

@dataclass
class LegalStatement:
    """Represents a legal statement with metadata"""
    original_text: str
    atomic_statements: List[str]
    statement_type: StatementType
    section_reference: str
    confidence_score: float

@dataclass
class IGCoding:
    """Represents institutional grammar coding for a statement"""
    statement_id: str
    statement_text: str
    description: str
    attribute: str = ""
    deontic: str = ""
    deontic_category: str = ""
    aim: str = ""
    object: str = ""
    activation_condition: str = ""
    execution_constraint: str = ""
    or_else: str = ""
    statement_type: str = ""
    confidence_score: float = 0.0
    confidence_reasoning: str = ""

class ProcessingState(TypedDict):
    """State schema for the LangGraph workflow"""
    document_text: str
    legal_statements: List[LegalStatement]
    atomic_statements: List[str]
    ig_codings: List[IGCoding]
    current_step: str
    progress: float
    errors: List[str]
    gemini_api_key: str
    progress_messages: List[str]

class DocumentProcessorAgent:
    """Agent responsible for initial document processing and legal statement extraction"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash-latest', generation_config={"response_mime_type": "application/json"})
    
    def _chunk_text(self, text: str, chunk_size: int = 7500, overlap: int = 500) -> List[str]:
        """Splits text into overlapping chunks to manage large documents."""
        if len(text) <= chunk_size:
            return [text]
        
        chunks = []
        start = 0
        while start < len(text):
            end = start + chunk_size
            chunks.append(text[start:end])
            start += chunk_size - overlap
        return chunks

    def extract_text_from_pdf(self, pdf_bytes: bytes) -> str:
        """Extract text from PDF using PyMuPDF"""
        try:
            doc = fitz.open(stream=pdf_bytes, filetype="pdf")
            text = ""
            for page in doc:
                text += page.get_text()
            doc.close()
            return text
        except Exception as e:
            logger.error(f"Error extracting text from PDF: {e}")
            return ""
    
    def identify_legal_statements(self, text: str) -> List[LegalStatement]:
        """Use Gemini to identify and extract legal statements from document text by processing it in chunks."""
        
        all_statements = []
        seen_statements = set()
        
        text_chunks = self._chunk_text(text)
        total_chunks = len(text_chunks)
        
        logger.info(f"Starting processing of {total_chunks} text chunks.")
        
        for i, chunk in enumerate(text_chunks):
            logger.info(f"Processing chunk {i+1}/{total_chunks}...")
            
            prompt = f"""
            You are an expert legal text analyst specializing in institutional analysis. 
            Analyze the following legal document text chunk and identify all institutional statements 
            that contain behavioral prescriptions, permissions, or prohibitions.
            
            Focus on statements that:
            1. Contain deontic operators (shall, must, may, should, etc.)
            2. Define rights, obligations, or procedures
            3. Establish institutional arrangements or rules
            4. Create or modify legal entities or relationships
            
            For each legal statement identified, provide:
            - The complete original text
            - Section reference if available
            - Whether it's regulative (prescribes behavior) or constitutive (defines entities/procedures)
            - Confidence score (0-1)
            
            Document text chunk:
            {chunk}
            
            Return your analysis in JSON format with the following structure:
            {{
                "legal_statements": [
                    {{
                        "original_text": "complete statement text",
                        "section_reference": "section number or reference",
                        "statement_type": "regulative|constitutive|hybrid",
                        "confidence_score": 0.95
                    }}
                ]
            }}
            """
            
            try:
                response = self.model.generate_content(prompt)
                result = json.loads(response.text)
                
                chunk_statements = result.get('legal_statements', [])
                if not chunk_statements:
                    logger.warning(f"No statements found in chunk {i+1}.")
                    continue
                    
                for stmt_data in chunk_statements:
                    original_text = stmt_data.get('original_text', '').strip()
                    if original_text and original_text not in seen_statements:
                        statement = LegalStatement(
                            original_text=original_text,
                            atomic_statements=[original_text],
                            statement_type=StatementType(stmt_data['statement_type']),
                            section_reference=stmt_data.get('section_reference', ''),
                            confidence_score=stmt_data.get('confidence_score', 0.8)
                        )
                        all_statements.append(statement)
                        seen_statements.add(original_text)
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON Decode Error in legal statement identification for chunk {i+1}: {e}")
                continue # Skip to the next chunk on error
            except Exception as e:
                logger.error(f"Error processing chunk {i+1}: {e}")
                continue # Skip to the next chunk on error

        logger.info(f"Finished processing all chunks. Found {len(all_statements)} unique statements.")
        return all_statements

class AtomicStatementAgent:
    """Agent responsible for converting legal statements into atomic statements"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash-latest', generation_config={"response_mime_type": "application/json"})
    
    def segment_to_atomic_statements(self, legal_statements: List[LegalStatement]) -> List[LegalStatement]:
        """Break down complex legal statements into atomic statements"""
        
        refined_statements = []
        
        for statement in legal_statements:
            prompt = f"""
            You are an expert in legal text analysis and institutional grammar.
            Your task is to take a single complex legal statement and break it down into multiple, simple atomic statements.
            Each atomic statement must contain only ONE institutional directive or rule.

            Rules for atomic statements:
            1. Each should have one clear subject (actor/attribute).
            2. One deontic operator (may, shall, must, etc.) if prescriptive.
            3. One main action or institutional function.
            4. Split compound sentences with multiple obligations into separate atomic statements.
            5. Preserve the legal meaning and context of the original statement.

            Example:
            Original statement: "The Data Fiduciary must, prior to or at the time of collection of personal data, give to the Data Principal a notice, and the Data Fiduciary shall not collect personal data unless it has obtained the consent of the Data Principal."
            JSON Output:
            {{
                "atomic_statements": [
                    "The Data Fiduciary must give to the Data Principal a notice prior to or at the time of collection of personal data.",
                    "The Data Fiduciary shall not collect personal data unless it has obtained the consent of the Data Principal."
                ]
            }}

            Now, process the following statement.

            Original statement:
            "{statement.original_text}"

            Return your result as a JSON object with a single key "atomic_statements" containing a list of strings.
            """
            
            try:
                response = self.model.generate_content(prompt)
                result = json.loads(response.text)
                
                atomic_texts = result.get('atomic_statements', [statement.original_text])
                
                refined_statement = LegalStatement(
                    original_text=statement.original_text,
                    atomic_statements=atomic_texts,
                    statement_type=statement.statement_type,
                    section_reference=statement.section_reference,
                    confidence_score=statement.confidence_score
                )
                refined_statements.append(refined_statement)
                
            except json.JSONDecodeError as e:
                logger.error(f"JSON Decode Error in atomic statement segmentation: {e}\\nRaw response:\\n{response.text}")
                refined_statements.append(statement) # Keep original on error
            except Exception as e:
                logger.error(f"Error in atomic statement segmentation: {e}")
                refined_statements.append(statement)
        
        return refined_statements

class IGCodingAgent:
    """Agent responsible for applying Institutional Grammar coding to atomic statements"""
    
    def __init__(self, api_key: str):
        genai.configure(api_key=api_key)
        self.model = genai.GenerativeModel('gemini-1.5-flash-latest', generation_config={"response_mime_type": "application/json"})
        
        # IG 2.0 component definitions based on research
        self.ig_components = {
            "Attribute": "The actor (individual or corporate) that carries out or is expected to carry out the action",
            "Deontic": "Prescriptive operator (may, shall, must, should, etc.) defining compulsion/restraint",
            "Aim": "The goal or action of the statement assigned to the attribute",
            "Object": "The receiver of the action (direct or indirect object)",
            "Activation_Condition": "Context that activates the statement (when/if conditions)",
            "Execution_Constraint": "Context that qualifies how action is performed",
            "Or_Else": "Consequence of violating the action specified in the aim"
        }
    
    def code_statements(self, refined_statements: List[LegalStatement]) -> List[IGCoding]:
        """Apply IG 2.0 coding to each atomic statement"""
        
        ig_codings = []
        statement_counter = 1
        
        for statement in refined_statements:
            for atomic_text in statement.atomic_statements:
                ig_coding = self._code_single_statement(
                    atomic_text, 
                    statement.section_reference,
                    statement_counter
                )
                ig_codings.append(ig_coding)
                statement_counter += 1
        
        return ig_codings
    
    def _code_single_statement(self, statement_text: str, section_ref: str, stmt_id: int) -> IGCoding:
        """Apply IG coding to a single atomic statement"""
        
        prompt = f"""
        You are a world-class expert in Institutional Grammar 2.0 (IG 2.0), tasked with precisely coding a single, atomic legal statement. Your analysis must be meticulous and adhere strictly to the IG 2.0 Codebook specifications.

        **Your Task:**
        1.  **Analyze the Statement**: Carefully read the atomic legal statement provided.
        2.  **Classify Statement Type**: Determine if it is 'regulative' or 'constitutive'.
        3.  **Extract Components**: Identify each IG 2.0 component based on the detailed definitions below.
        4.  **Provide Reasoning**: Explain your confidence score and the key reasons for your coding decisions.
        5.  **Format Output**: Return a single, valid JSON object with all fields filled.

        **--- IG 2.0 Component Definitions ---**

        *   **Attribute (A)**: The actor (individual or corporate) that carries out, or is expected to/to not carry out, the action (i.e., Aim) of the statement.
        *   **Deontic (D)**: The prescriptive operator defining compulsion or restraint (e.g., must, shall, may, is prohibited).
        *   **Deontic Category**: 'O' (Obligation), 'P' (Permission), or 'F' (Prohibition).
        *   **Aim (I)**: The goal or action of the statement assigned to the Attribute.
        *   **Object (B)**: The inanimate or animate receiver of the action in the Aim. Can be direct or indirect.
        *   **Context (Cac & Cex)**:
            *   **Activation Condition (Cac)**: Instantiates the setting or event that *activates* the entire rule. **Heuristic**: Ask "Does this clause set a condition *for the rule to apply*?" (e.g., "Upon receiving final notice...", "Starting January 1...").
            *   **Execution Constraint (Cex)**: Qualifies *how* the action is performed once activated. **Heuristic**: Ask "Does this clause qualify the *manner, means, or purpose* of the action?" (e.g., "...by regulations", "...annually", "in the farmer's market").
        *   **Or Else (O)**: The consequence of violating the action specified in the Aim.
        *   **Description**: A brief, clear summary of the institutional rule.
        *   **Statement Type**: 'regulative' or 'constitutive'.
        *   **Confidence Score**: A float from 0.0 to 1.0 indicating your confidence in the coding accuracy.
        *   **Confidence Reasoning**: A brief explanation for your confidence score and coding choices.

        **--- High-Quality Example ---**

        Statement: "Upon entrance into agreement with an organic farmer to serve as his/her certifying agent, the organic certifier must inspect the farmer's operation within 60 days."

        JSON Output:
        {{
            "attribute": "the organic certifier",
            "deontic": "must",
            "deontic_category": "O",
            "aim": "inspect the farmer's operation",
            "object": "the farmer's operation",
            "activation_condition": "Upon entrance into agreement with an organic farmer to serve as his/her certifying agent",
            "execution_constraint": "within 60 days",
            "or_else": "",
            "description": "A certifier is obligated to inspect a farmer's operation within a specific timeframe after an agreement is made.",
            "statement_type": "regulative",
            "confidence_score": 0.98,
            "confidence_reasoning": "The statement is clearly regulative. The 'Activation Condition' is a procedural event that triggers the rule. The 'Execution Constraint' is a temporal constraint ('timeframe') on the action."
        }}

        **--- Your Turn ---**

        Statement to Code: "{statement_text}"

        Provide your analysis in the JSON format specified above. Ensure every field is filled.
        """
        
        try:
            response = self.model.generate_content(prompt)
            result = json.loads(response.text)
            
            # --- Self-Validation Step ---
            confidence = result.get('confidence_score', 0.8)
            if not result.get('attribute') or not result.get('aim'):
                confidence *= 0.75 # Penalize confidence if critical fields are missing
                logger.warning(f"Low quality IG coding for statement: {{statement_text}}. Missing Attribute or Aim.")

            return IGCoding(
                statement_id=f"{section_ref}({stmt_id})" if section_ref else str(stmt_id),
                statement_text=statement_text,
                description=result.get('description', ''),
                attribute=result.get('attribute', ''),
                deontic=result.get('deontic', ''),
                deontic_category=result.get('deontic_category', ''),
                aim=result.get('aim', ''),
                object=result.get('object', ''),
                activation_condition=result.get('activation_condition', ''),
                execution_constraint=result.get('execution_constraint', ''),
                or_else=result.get('or_else', ''),
                statement_type=result.get('statement_type', 'regulative'),
                confidence_score=confidence,
                confidence_reasoning=result.get('confidence_reasoning', '')
            )
            
        except json.JSONDecodeError as e:
            logger.error(f"JSON Decode Error in IG coding: {e}\\nRaw response:\\n{response.text}")
            return IGCoding(
                statement_id=f"{section_ref}({stmt_id})" if section_ref else str(stmt_id),
                statement_text=statement_text,
                description="Error: Failed to parse model output",
                statement_type="unknown"
            )
        except Exception as e:
            logger.error(f"Error in IG coding: {e}")
            return IGCoding(
                statement_id=f"{section_ref}({stmt_id})" if section_ref else str(stmt_id),
                statement_text=statement_text,
                description="Error: Failed to parse model output",
                statement_type="unknown"
            )

class WorkflowOrchestrator:
    """LangGraph-based workflow orchestrator for the multi-agent system"""
    
    def __init__(self, api_key: str):
        self.api_key = api_key
        self.document_processor = DocumentProcessorAgent(api_key)
        self.atomic_agent = AtomicStatementAgent(api_key)
        self.ig_agent = IGCodingAgent(api_key)
        
        # Create the workflow graph
        self.workflow = self._create_workflow()
    
    def _create_workflow(self) -> StateGraph:
        """Create the LangGraph workflow"""
        
        workflow = StateGraph(ProcessingState)
        
        # Add nodes
        workflow.add_node("extract_legal_statements", self._extract_legal_statements_node)
        workflow.add_node("create_atomic_statements", self._create_atomic_statements_node)
        workflow.add_node("apply_ig_coding", self._apply_ig_coding_node)
        workflow.add_node("finalize_results", self._finalize_results_node)
        
        # Add edges
        workflow.add_edge(START, "extract_legal_statements")
        workflow.add_edge("extract_legal_statements", "create_atomic_statements")
        workflow.add_edge("create_atomic_statements", "apply_ig_coding")
        workflow.add_edge("apply_ig_coding", "finalize_results")
        workflow.add_edge("finalize_results", END)
        
        # Set entry point
        workflow.set_entry_point("extract_legal_statements")
        
        return workflow.compile(checkpointer=MemorySaver())
    
    def _extract_legal_statements_node(self, state: ProcessingState) -> ProcessingState:
        """Node for extracting legal statements"""
        try:
            state["progress_messages"].append("🔍 Extracting legal statements from document...")
            
            legal_statements = self.document_processor.identify_legal_statements(state["document_text"])
            
            state["legal_statements"] = legal_statements
            state["current_step"] = "Legal statements extracted"
            state["progress"] = 25.0
            
            state["progress_messages"].append(f"✅ Found {len(legal_statements)} legal statements")
            
        except Exception as e:
            state["errors"].append(f"Error in legal statement extraction: {str(e)}")
            logger.error(f"Error in extract_legal_statements_node: {e}")
        
        return state
    
    def _create_atomic_statements_node(self, state: ProcessingState) -> ProcessingState:
        """Node for creating atomic statements"""
        try:
            state["progress_messages"].append("⚛️ Segmenting into atomic statements...")
            
            refined_statements = self.atomic_agent.segment_to_atomic_statements(state["legal_statements"])
            
            # Collect all atomic statements
            atomic_statements = []
            for stmt in refined_statements:
                atomic_statements.extend(stmt.atomic_statements)
            
            state["legal_statements"] = refined_statements
            state["atomic_statements"] = atomic_statements
            state["current_step"] = "Atomic statements created"
            state["progress"] = 50.0
            
            state["progress_messages"].append(f"✅ Created {len(atomic_statements)} atomic statements")
            
        except Exception as e:
            state["errors"].append(f"Error in atomic statement creation: {str(e)}")
            logger.error(f"Error in create_atomic_statements_node: {e}")
        
            return state

    def _apply_ig_coding_node(self, state: ProcessingState) -> ProcessingState:
        """Node for applying IG coding"""
        try:
            state["progress_messages"].append("📝 Applying Institutional Grammar coding...")
            
            ig_codings = self.ig_agent.code_statements(state["legal_statements"])
            
            state["ig_codings"] = ig_codings
            state["current_step"] = "IG coding applied"
            state["progress"] = 85.0
            
            state["progress_messages"].append(f"✅ Applied IG coding to {len(ig_codings)} statements")
            
        except Exception as e:
            state["errors"].append(f"Error in IG coding: {str(e)}")
            logger.error(f"Error in apply_ig_coding_node: {e}")
        
        return state
    
    def _finalize_results_node(self, state: ProcessingState) -> ProcessingState:
        """Node for finalizing results"""
        try:
            state["progress_messages"].append("📊 Finalizing results...")
            
            state["current_step"] = "Processing complete"
            state["progress"] = 100.0
            
            state["progress_messages"].append("✅ Processing completed successfully!")
            
        except Exception as e:
            state["errors"].append(f"Error in finalization: {str(e)}")
            logger.error(f"Error in finalize_results_node: {e}")
        
        return state
    
    def process_document(self, document_text: str, api_key: str, progress_messages: List[str]) -> ProcessingState:
        """Process document through the complete workflow"""
        
        initial_state = ProcessingState(
            document_text=document_text,
            legal_statements=[],
            atomic_statements=[],
            ig_codings=[],
            current_step="Starting processing",
            progress=0.0,
            errors=[],
            gemini_api_key=api_key,
            progress_messages=progress_messages
        )
        
        # Define a configurable for the checkpointer to track the conversation
        config = {"configurable": {"thread_id": str(uuid.uuid4())}}
        
        # Run the workflow
        result = self.workflow.invoke(initial_state, config=config)
        return result

def create_excel_output(ig_codings: List[IGCoding]) -> io.BytesIO:
    """Create Excel file from IG codings matching the example format"""
    
    # Create DataFrame with the specified columns
    data = []
    for coding in ig_codings:
        data.append({
            'Description': coding.description,
            'Statement_ID': coding.statement_id,
            'Statement': coding.statement_text,
            'Attribute[A]': coding.attribute,
            'Deontic[D]': coding.deontic,
            'Deontic_Category': coding.deontic_category,
            'Aim[I]': coding.aim,
            'Object[B]': coding.object,
            'Activation_Condition[Cac]': coding.activation_condition,
            'Execution_Constraint[Cex]': coding.execution_constraint,
            'Or_Else[O]': coding.or_else,
            'Confidence_Score': coding.confidence_score,
            'Confidence_Reasoning': coding.confidence_reasoning
        })
    
    df = pd.DataFrame(data)
    
    # Create Excel file
    excel_buffer = io.BytesIO()
    with pd.ExcelWriter(excel_buffer, engine='openpyxl') as writer:
        df.to_excel(writer, sheet_name='IG_Coding', index=False)
        
        # Access the workbook and worksheet
        workbook = writer.book
        worksheet = writer.sheets['IG_Coding']
        
        # Apply formatting
        header_font = Font(bold=True, color='FFFFFF')
        header_fill = PatternFill(start_color='4472C4', end_color='4472C4', fill_type='solid')
        
        # Format headers
        for cell in worksheet[1]:
            cell.font = header_font
            cell.fill = header_fill
            cell.alignment = Alignment(horizontal='center', vertical='center')
        
        # Auto-adjust column widths
        for column in worksheet.columns:
            max_length = 0
            column_letter = column[0].column_letter
            for cell in column:
                try:
                    if len(str(cell.value)) > max_length:
                        max_length = len(str(cell.value))
                except:
                    pass
            adjusted_width = min(max_length + 2, 50)
            worksheet.column_dimensions[column_letter].width = adjusted_width
    
    excel_buffer.seek(0)
    return excel_buffer

def main():
    """Main Streamlit application"""
    
    st.set_page_config(
        page_title="Institutional Grammar Legal Document Processor",
        page_icon="⚖️",
        layout="wide"
    )
    
    st.title("⚖️ Institutional Grammar Legal Document Processor")
    st.markdown("### Multi-Agent System for Automated IG Coding of Legal Documents")
    
    # Sidebar for API configuration
    st.sidebar.header("🔧 Configuration")
    api_key = st.sidebar.text_input(
        "Gemini API Key", 
        type="password",
        help="Enter your Google Gemini API key. Get one at https://ai.google.dev/"
    )
    
    if not api_key:
        st.warning("⚠️ Please enter your Gemini API key in the sidebar to continue.")
        st.info("""
        **To get started:**
        1. Get a Gemini API key from https://ai.google.dev/
        2. Enter the API key in the sidebar
        3. Upload a legal document (PDF)
        4. Click 'Process Document' to begin analysis
        """)
        return
    
    # Define a callback function to handle file uploads
    def handle_pdf_upload():
        uploaded_file = st.session_state.get("pdf_file_uploader")
        if uploaded_file:
            try:
                with st.spinner(f"📖 Reading PDF: {uploaded_file.name}..."):
                    temp_processor = DocumentProcessorAgent(api_key)
                    extracted_text = temp_processor.extract_text_from_pdf(uploaded_file.getvalue())
                    if extracted_text and extracted_text.strip():
                        st.session_state.document_text_content = extracted_text
                        st.success(f"✅ Extracted {len(extracted_text)} characters. Review in the 'Paste / Review Text' tab.")
                    else:
                        st.session_state.document_text_content = ""
                        st.warning("⚠️ Could not extract readable text from the PDF.")
            except Exception as e:
                st.session_state.document_text_content = ""
                st.error(f"❌ Failed to read PDF: {str(e)}")

    # Document Input Section
    st.header("📄 Document Input")
    tab1, tab2 = st.tabs(["Upload PDF File", "Paste / Review Text"])

    with tab1:
        st.file_uploader(
            "Upload a legal document (PDF)",
            type=['pdf'],
            help="Upload a PDF. The extracted text will appear in the 'Paste / Review Text' tab.",
            label_visibility="collapsed",
            key="pdf_file_uploader",
            on_change=handle_pdf_upload
        )

    with tab2:
        st.text_area(
            "The content of this text area will be processed. You can paste text directly or upload a PDF to populate it.",
            height=300,
            placeholder="Upload a PDF or paste content here...",
            key="document_text_content"
        )
    
    if st.button("🚀 Process Document", type="primary"):
        document_text = st.session_state.get("document_text_content", "")

        if not document_text or not document_text.strip():
            st.warning("⚠️ The text box is empty. Please upload a file or paste text to process.")
            return

        st.success(f"✅ Processing {len(document_text)} characters from the text area.")
        
        # Common processing logic for the extracted text
        # Clear previous results
        st.session_state.processing_complete = False
        st.session_state.results_df = None
        st.session_state.progress_messages = []

        try:
            orchestrator = WorkflowOrchestrator(api_key)
            
            # Create progress containers
            progress_bar = st.progress(0)
            status_container = st.container()
            
            # Process document in a background thread
            with st.spinner("🔄 Processing document through multi-agent workflow..."):
                with ThreadPoolExecutor() as executor:
                    future = executor.submit(
                        orchestrator.process_document,
                        document_text, 
                        api_key, 
                        st.session_state.progress_messages
                    )
                    
                    # Update progress UI while processing
                    while not future.done():
                        status_container.empty()
                        with status_container:
                            for msg in st.session_state.progress_messages:
                                st.text(msg)
                        time.sleep(0.5)
                    
                    result = future.result()

            progress_bar.progress(1.0)
            
            # Display final status messages
            status_container.empty()
            with status_container:
                for msg in st.session_state.progress_messages:
                    st.text(msg)

            # Check for errors from the workflow
            if result["errors"]:
                st.error("❌ Errors occurred during processing:")
                for error in result["errors"]:
                    st.error(f"• {error}")
                return

            # Create and store results DataFrame
            if result["ig_codings"]:
                st.session_state.results_df = pd.DataFrame([
                    {
                        'Description': coding.description,
                        'Statement_ID': coding.statement_id,
                        'Statement': coding.statement_text,
                        'Attribute[A]': coding.attribute,
                        'Deontic[D]': coding.deontic,
                        'Deontic_Category': coding.deontic_category,
                        'Aim[I]': coding.aim,
                        'Object[B]': coding.object,
                        'Activation_Condition[Cac]': coding.activation_condition,
                        'Execution_Constraint[Cex]': coding.execution_constraint,
                        'Or_Else[O]': coding.or_else,
                        'Confidence_Score': coding.confidence_score,
                        'Confidence_Reasoning': coding.confidence_reasoning
                    }
                    for coding in result["ig_codings"]
                ])
                st.session_state.processing_complete = True
                st.success("✅ Processing completed successfully!")
            else:
                st.warning("⚠️ No institutional grammar codings were generated.")

        except Exception as e:
            st.error(f"❌ An unexpected error occurred: {str(e)}")
            logger.error(f"Error in main processing: {e}")
    
    # Display results if processing is complete
    if st.session_state.processing_complete and st.session_state.results_df is not None:
        
        st.header("📊 Results")
        
        # Show summary statistics
        col1, col2, col3, col4 = st.columns(4)
        with col1:
            st.metric("Total Statements", len(st.session_state.results_df))
        with col2:
            regulative_count = len(st.session_state.results_df[st.session_state.results_df['Deontic[D]'].notna() & (st.session_state.results_df['Deontic[D]'] != '')])
            st.metric("Regulative Statements", regulative_count)
        with col3:
            st.metric("Components Identified", st.session_state.results_df.notna().sum().sum())
        with col4:
            avg_confidence = st.session_state.results_df['Confidence_Score'].mean()
            st.metric("Avg. Confidence", f"{avg_confidence:.2f}")
        
        # Display results table
        st.subheader("📋 Institutional Grammar Coding Results")
        st.dataframe(
            st.session_state.results_df,
            use_container_width=True,
            height=400
        )
        
        # Download button
        st.subheader("💾 Download Results")
        
        try:
            excel_buffer = create_excel_output([
                IGCoding(
                    statement_id=row['Statement_ID'],
                    statement_text=row['Statement'],
                    description=row['Description'],
                    attribute=row['Attribute[A]'],
                    deontic=row['Deontic[D]'],
                    deontic_category=row['Deontic_Category'],
                    aim=row['Aim[I]'],
                    object=row['Object[B]'],
                    activation_condition=row['Activation_Condition[Cac]'],
                    execution_constraint=row['Execution_Constraint[Cex]'],
                    or_else=row['Or_Else[O]'],
                    confidence_score=row['Confidence_Score'],
                    confidence_reasoning=row['Confidence_Reasoning']
                )
                for _, row in st.session_state.results_df.iterrows()
            ])
            
            st.download_button(
                label="📥 Download Excel File",
                data=excel_buffer.getvalue(),
                file_name=f"institutional_grammar_coding_{int(time.time())}.xlsx",
                mime="application/vnd.openxmlformats-officedocument.spreadsheetml.sheet",
                type="primary"
            )
            
        except Exception as e:
            st.error(f"❌ Error creating Excel file: {str(e)}")
    
    # Footer
    st.markdown("---")
    st.markdown("""
    **About this tool:**
    This application uses advanced AI agents orchestrated through LangGraph to automatically process legal documents 
    and apply Institutional Grammar 2.0 coding. The system leverages Google's Gemini API for natural language 
    understanding and follows established patterns for legal document analysis.
    
    **Technologies used:** LangGraph, Gemini API, Streamlit, PyMuPDF, spaCy, pandas, openpyxl
    """)

if __name__ == "__main__":
    main()
