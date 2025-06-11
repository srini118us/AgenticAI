def validate_output(output: str, validation_criteria: str = "not_empty") -> bool:
    """Performs a simple validation check on the given output."""
    if validation_criteria == "not_empty":
        return bool(output and output.strip())
    elif validation_criteria == "contains_keyword":
        # Example: check if output contains a specific keyword (needs a keyword to be passed)
        # For this example, let's just check if it contains "valid"
        return "valid" in output.lower()
    else:
        print(f"Warning: Unknown validation criteria '{validation_criteria}'. Defaulting to not_empty.")
        return bool(output and output.strip())

if __name__ == "__main__":
    print("--- Validator Simple Example ---")
    
    # Test 1: Not empty
    output1 = "This is a valid output."
    is_valid1 = validate_output(output1)
    print(f"\"'{output1}'\" is valid (not_empty): {is_valid1}")
    
    # Test 2: Empty
    output2 = "  "
    is_valid2 = validate_output(output2)
    print(f"\"'{output2}'\" is valid (not_empty): {is_valid2}")
    
    # Test 3: Contains keyword
    output3 = "The result is valid."
    is_valid3 = validate_output(output3, validation_criteria="contains_keyword")
    print(f"\"'{output3}'\" is valid (contains_keyword 'valid'): {is_valid3}")
    
    output4 = "The result is not good."
    is_valid4 = validate_output(output4, validation_criteria="contains_keyword")
    print(f"\"'{output4}'\" is valid (contains_keyword 'valid'): {is_valid4}") 