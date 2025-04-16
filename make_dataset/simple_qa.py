import sys
import os
import pandas as pd
import json
import re  # Added for pattern matching
from typing import Dict, List

# Assuming these imports still exist and work the same way
current_dir = os.path.dirname(os.path.abspath(__file__))
root_dir = os.path.dirname(current_dir)
sys.path.append(root_dir)
from src.utils.config import load_config
from make_dataset.models import ChatMessage, CutMessage, skip_type_list
from make_dataset.strategies import TimeWindowStrategy, LLMStrategy

class SimpleDataProcessor:
    def __init__(self):
        self.config = load_config(arg_type="make_dataset")
        self.csv_folder = "./data/csv"
        
        # Keep the same strategy initialization
        if self.config["single_combine_strategy"] == "time_window":
            self.single_combine_strategy = TimeWindowStrategy(
                time_window=self.config["single_combine_time_window"] * 60,
                is_single_chat=True,
            )
        else:
            self.single_combine_strategy = LLMStrategy(is_single_chat=True)
            
        if self.config["qa_match_strategy"] == "time_window":
            self.qa_match_strategy = TimeWindowStrategy(
                time_window=self.config["qa_match_time_window"] * 60,
                is_single_chat=False,
            )
        else:
            self.qa_match_strategy = LLMStrategy(is_single_chat=False)
            
        self.c = self.config
        
        # Add statistics to track filtered messages
        self.stats = {
            "total_messages": 0,
            "filtered_messages": 0,
            "total_qa_pairs": 0,
            "filtered_qa_pairs": 0
        }

    def get_csv_files(self):
        """Get all CSV files from the folder"""
        csv_files = []
        for chat_obj_folder in os.listdir(self.csv_folder):
            chat_obj_folder_path = os.path.join(self.csv_folder, chat_obj_folder)
            if os.path.isdir(chat_obj_folder_path):
                for csvfile in os.listdir(chat_obj_folder_path):
                    if csvfile.endswith(".csv"):
                        csvfile_path = os.path.join(chat_obj_folder_path, csvfile)
                        csv_files.append(csvfile_path)
        return csv_files
    
    def contains_bracket_text(self, text):
        """
        Check if the message contains text within square brackets like [表情包消息], [图片消息], etc.
        
        Args:
            text (str): The message text to check
            
        Returns:
            bool: True if the text contains patterns within square brackets, False otherwise
        """
        if not text or not isinstance(text, str):
            return False
        
        # Pattern to match any text inside square brackets
        pattern = r'\[.*?\]'
        return bool(re.search(pattern, text))

    def main(self):
        csv_files = self.get_csv_files()
        message_list = []
        
        for csv_file in csv_files:
            print(f"Processing {csv_file}")
            chat_messages = self.simple_load_csv(csv_file)
            message_list.extend(chat_messages)
        
        self.stats["total_messages"] = len(message_list)    
        qa_res = self.match_qa(message_list)
        
        if self.c["prompt_with_history"]:
            qa_res = self.add_history_to_qa(qa_res)
            
        self.save_result(qa_res)
        
        # Print summary statistics
        print(f"Processed {self.stats['total_messages']} messages")
        print(f"Filtered out {self.stats['filtered_messages']} messages containing bracket text")
        print(f"Generated {self.stats['total_qa_pairs']} QA pairs")
        print(f"Filtered out {self.stats['filtered_qa_pairs']} QA pairs containing bracket text")
        print(f"Final output: {len(qa_res)} QA pairs")

    def simple_load_csv(self, file_path) -> List[ChatMessage]:
        """Simple CSV loader without extensive checks"""
        try:
            # Read CSV file
            df = pd.read_csv(file_path, encoding="utf-8")
            
            # Add missing columns if needed
            required_columns = ["id", "MsgSvrID", "type_name", "is_sender", "talker", "room_name", "content", "CreateTime"]
            for col in required_columns:
                if col not in df.columns:
                    df[col] = "" if col != "CreateTime" else pd.Timestamp.now()
            
            # Extract msg and src from content
            df["msg"] = ""
            df["src"] = ""
            
            for i in df.index:
                if df.loc[i, "type_name"] == "文本" and not pd.isna(df.loc[i, "content"]):
                    try:
                        content_str = str(df.loc[i, "content"])
                        parsed_data = json.loads(content_str)
                        df.loc[i, "msg"] = parsed_data.get("msg", "")
                        df.loc[i, "src"] = parsed_data.get("src", "")
                    except:
                        # If parsing fails, use the content as msg
                        df.loc[i, "msg"] = str(df.loc[i, "content"])
            
            # Ensure CreateTime is a timestamp
            df["CreateTime"] = pd.to_datetime(df["CreateTime"], errors='coerce')
            df["CreateTime"] = df["CreateTime"].fillna(pd.Timestamp.now())
            
            # Select only the columns needed for ChatMessage
            columns = ["id", "MsgSvrID", "type_name", "is_sender", "talker", "room_name", "msg", "src", "CreateTime"]
            selected_df = df[columns]
            
            # Convert to ChatMessage objects
            chat_messages = []
            for _, row in selected_df.iterrows():
                try:
                    chat_messages.append(ChatMessage(
                        id=row["id"],
                        MsgSvrID=row["MsgSvrID"],
                        type_name=row["type_name"],
                        is_sender=row["is_sender"],
                        talker=row["talker"],
                        room_name=row["room_name"],
                        msg=row["msg"],
                        src=row["src"],
                        CreateTime=row["CreateTime"]
                    ))
                except Exception as e:
                    print(f"Error creating ChatMessage: {e}")
            
            return chat_messages
            
        except Exception as e:
            print(f"Error processing file {file_path}: {e}")
            return []

    def match_qa(self, messages):
        """
        Match questions and answers from messages while filtering out pairs 
        containing bracketed text like [表情包消息], [图片消息], etc.
        """
        WAITING_INSTRUCTION = "waiting_instruction"
        WAITING_RESPONSE = "waiting_response"

        current_state = WAITING_INSTRUCTION
        qa_res = []
        last_message = None
        current_instruction = None

        for msg in messages:
            # Count messages with bracket text
            if isinstance(msg, ChatMessage) and self.contains_bracket_text(msg.msg):
                self.stats["filtered_messages"] += 1
                
            if isinstance(msg, CutMessage):
                current_state = WAITING_INSTRUCTION
                current_instruction = None
                last_message = None
                if self.c["prompt_with_history"]:
                    qa_res.append(msg)
                continue

            if current_state == WAITING_INSTRUCTION:
                if msg.is_sender == 0:  # Received message from other party
                    current_instruction = msg.msg
                    last_message = msg
                    current_state = WAITING_RESPONSE

            elif current_state == WAITING_RESPONSE:
                if msg.is_sender == 0:  # Received message from other party
                    current_instruction = msg.msg
                    last_message = msg
                    # State remains unchanged
                else:  # Own reply
                    if last_message and self.qa_match_strategy.is_same_conversation([last_message], msg):
                        # Check if either instruction or response contains bracket text
                        if (not self.contains_bracket_text(current_instruction) and 
                            not self.contains_bracket_text(msg.msg)):
                            qa_res.append({"instruction": current_instruction, "output": msg.msg})
                            self.stats["total_qa_pairs"] += 1
                        else:
                            # Skip this QA pair as it contains bracketed text
                            self.stats["filtered_qa_pairs"] += 1
                    else:
                        if self.c["prompt_with_history"]:
                            qa_res.append(CutMessage(
                                is_sender=msg.is_sender,
                                cut_type=msg.type_name,
                                CreateTime=msg.CreateTime,
                            ))
                    # Reset state
                    current_state = WAITING_INSTRUCTION
                    current_instruction = None
                    last_message = None

        return qa_res

    def add_history_to_qa(self, qa_res):
        """
        Add conversation history to QA pairs and filter out any history containing bracketed text
        """
        qa_res_with_history = []
        last_res = {"instruction": "", "output": "", "history": []}

        for qa in qa_res:
            if isinstance(qa, CutMessage):
                if len(last_res["history"]) == 0:
                    continue
                else:
                    if len(last_res["history"]) == 1:
                        # Simple case with just one history item
                        last_res = {
                            "instruction": last_res["history"][0][0],
                            "output": last_res["history"][0][1],
                            "history": [],
                        }
                        qa_res_with_history.append(last_res)
                    else:
                        # Filter history to remove any pairs with bracket text
                        filtered_history = []
                        for h_instruction, h_output in last_res["history"][:-1]:
                            if (not self.contains_bracket_text(h_instruction) and 
                                not self.contains_bracket_text(h_output)):
                                filtered_history.append([h_instruction, h_output])
                        
                        last_res = {
                            "instruction": last_res["history"][-1][0],
                            "output": last_res["history"][-1][1],
                            "history": filtered_history,
                        }
                        qa_res_with_history.append(last_res)
                    
                    last_res = {"instruction": "", "output": "", "history": []}
            else:
                # Only add to history if it doesn't contain bracketed text
                if (not self.contains_bracket_text(qa["instruction"]) and 
                    not self.contains_bracket_text(qa["output"])):
                    last_res["history"].append([qa["instruction"], qa["output"]])

        # Process any remaining history
        if len(last_res["history"]) > 0:
            if len(last_res["history"]) == 1:
                last_res = {
                    "instruction": last_res["history"][0][0],
                    "output": last_res["history"][0][1],
                    "history": [],
                }
                qa_res_with_history.append(last_res)
            else:
                # Filter history to remove any pairs with bracket text
                filtered_history = []
                for h_instruction, h_output in last_res["history"][:-1]:
                    if (not self.contains_bracket_text(h_instruction) and 
                        not self.contains_bracket_text(h_output)):
                        filtered_history.append([h_instruction, h_output])
                
                last_res = {
                    "instruction": last_res["history"][-1][0],
                    "output": last_res["history"][-1][1],
                    "history": filtered_history,
                }
                qa_res_with_history.append(last_res)

        return qa_res_with_history

    def save_result(self, qa_res):
        """Save the processed QA pairs to a JSON file"""
        output_dir = "./data/res_csv/sftx"
        os.makedirs(output_dir, exist_ok=True)
        output_file = os.path.join(output_dir, "sft-my-2.json")
        
        with open(output_file, "w", encoding="utf-8") as f:
            json.dump(qa_res, f, ensure_ascii=False, indent=2)
        
        print(f"Chat processing completed. {len(qa_res)} records saved to {output_file}")

if __name__ == "__main__":
    processor = SimpleDataProcessor()
    processor.main()