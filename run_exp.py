# Load existing conversation logs if the file exists
if os.path.exists(results_path):
    with open(results_path, "r") as f:
        existing_logs = json.load(f)
else:
    existing_logs = []

# Append the new conversation log to the existing logs
existing_logs.append(conversation_log)

# Save the updated conversation logs
try:
    with open(results_path, "w") as f:
        json.dump(existing_logs, f, indent=4)
    print("Conversation saved successfully.")
except Exception as e:
    print(f"An error occurred while saving the conversation: {e}")
