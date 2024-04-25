## Team: NLP WIZARDS

## Semantic Analysis and Role Labeling in News Articles
##NLP WIZARDS
## Team Members

This project is designed to automate the redaction of sensitive information from text documents to ensure privacy and adherence to data protection laws. Initially, the project scans files in specified directories to detect sensitive data, such as names, physical addresses, dates, and phone numbers. It then generates output files with a .censored extension, where all identified sensitive information is concealed. The project employs a combination of techniques for detecting sensitive data, including regex matching, Google NLP, and spaCy, ensuring a comprehensive and accurate identification process. Furthermore, it records statistics on the number of names, physical addresses, dates, and phone numbers identified and redacted, providing valuable insights into the extent of sensitive information contained within the documents.

Note: This project considers only names, phone numbers, and date as sensitive content to be hidden.

## How to install and run

1.Install pyenv and pipenv if not already installed.

2.Install the Python version of your choice using the command "pyenv install {PYTHON_VERSION}".

3.Open the project folder and set the local Python version with "pyenv local {PYTHON_VERSION}".

5.To create a pipenv environment for the project, run "pipenv install".

6.Activate the pipenv environment using the command "pipenv shell".

7.To run test cases and check the application's functionality, enter the command: 'pipenv run python -m pytest'.

8.Execute the application by typing: 'pipenv run python censoror.py --input '<glob pattern(ex: *.txt)>' --names --dates --phones --address --output '{output directory path}' --stats {stats file path}

Note: only --input, --names, --dates, --phones, --address, --output, --stats are supported parameters and are mandatory parameters to be passed

## Assumptions

1. API key is provided in the google_api_key.json amd has right permission to execute NLP operations.
2. Address and phone number should be in right format, else the methods used may not be abel to catch it.
3. Google NLP and Spacy are able to catch most of the sensitive data.

## Bugs

1. Any name not recognized as PERSON by spacy or google NLP will not be hidden.
2. Script will only hide US address in right format and locations which are identified by  spacy and google NLP.
3. Any date not recognized as DATE by spacy or google NLP will not be hidden.
4. Any phone numbers not identified by google NLP or regex will not be hidden.
5. Additional words may be hidden in the output as spacy and Google NLP may identify words which are not sensitive.


## Functions Description
**censoror.py**

**create_censored_file**

    Writes the censored version of a document to a new file in the specified output directory.

    This function takes the censored text content, generates a new filename by appending '.censored' 
    to the original filename, and writes the censored content to this new file in the given output directory.

    Parameters:
    - output_string (str): The censored text to be written to the new file.
    - file_path (str): The path to the original file. This is used to generate the new filename.
    - out_folder_path (str): The path to the output directory where the censored file will be saved.

    Returns:
    None
   

**parse_arguments**
     
    Parses command-line arguments for the Censoror script
    
    This function configures and parses command-line arguments required to run the script, including input file patterns,
    output directory, and flags for specifying types of sensitive information to censor (names, dates, phone numbers, addresses).
    It also handles the specification of where to output statistics about the censor process.

    Returns:
    argparse.Namespace: An object containing all the command-line arguments specified by the user. This includes:
        - input: Glob patterns for the files to be censored.
        - output: The directory where the censored files will be saved.
        - names: Flag to indicate whether names should be censored.
        - dates: Flag to indicate whether dates should be censored.
        - phones: Flag to indicate whether phone numbers should be censored.
        - address: Flag to indicate whether addresses should be censored.
        - stats: Specifies where to write the statistics (stderr, stdout, or a filename).
	
**write_stats**

    Writes the censor statistics to a specified destination.

    This function takes the compiled statistics from the censorship process and writes them to either a standard output 
    (stdout), standard error (stderr), or a specific file. The destination is determined by the user's input.

    Parameters:
    - stats (str): A string containing the compiled statistics of the censorship process.
    - destination (str): Specifies where to write the statistics. It can be 'stderr', 'stdout', or a file path. 
                         If a directory path is provided, the statistics will be written to a file named "stats" within that directory.

    Returns:
    None. This function directly writes the statistics to the specified destination without returning any value.

**main**

	Main function to execute the censorship process on text files.

    This function orchestrates the entire process of reading text files, identifying and censoring sensitive information
    (names, addresses, dates, and phone numbers), and then writing the censored content to new files. It leverages a 
    pre-trained spaCy model for NLP tasks and a custom class `MaskString` for masking the identified sensitive information.
    The function also compiles statistics about the censored information for each file and writes these statistics to the
    specified destination.

    Parameters:
    None. Command-line arguments are parsed within the function using `parse_arguments`.

    Returns:
    None. This function's primary purpose is to execute a series of operations for censoring text files and does not return any value.

    Raises:
    Exception: Catches and logs exceptions related to file processing, ensuring the program continues to process subsequent files.
	
**assignment1/hidestring.py**

***MaskString***

    A class for masking sensitive information in a given text string using various methods, including spaCy NLP and Google Cloud NLP API.

    Attributes:
        input_string (str): The input text string to be processed and censored.
        nlp (spaCy Language): A spaCy Language object loaded with the "en_core_web_md" model for natural language processing.
        doc (spaCy Doc): A spaCy document object, processed version of input_string, used for NLP tasks.
        partially_hidden_string (str): A version of input_string where sensitive information has been masked.
        gcp_entities (list): A list of entities recognized by Google Cloud NLP API in the input_string.

**set_input_string**

    Sets the input string to be processed for censoring sensitive information.

    Parameters:
        input_string (str): The text string to be analyzed and processed.

    This method updates the class attributes to prepare for the analysis and censoring process. It performs the following actions:
    - Sets the input_string attribute to the given input string.
    - Processes the input string with spaCy NLP to create a document object, which is then stored in the doc attribute.
    - Initializes the partially_hidden_string attribute with the input string, which will be modified as sensitive information is masked.
    - Invokes the gcp_nlp method to analyze entities within the input string using Google Cloud NLP, storing the results in the gcp_entities attribute.

**gcp_nlp**
    
    Analyzes the given input string using Google Cloud Natural Language Processing (NLP) to identify entities.

    Parameters:
        input_string (str): The text string to be analyzed by Google Cloud NLP.

    Returns:
        entities (list): A list of entities identified by Google Cloud NLP within the input string.

    This method sends a request to the Google Cloud Natural Language API with the input string to analyze entities present in the text. It constructs a request with the necessary headers and payload, specifying the content type and encoding type, along with the document content. Upon receiving the response, it extracts and returns the entities identified in the text.

    Note: The API key is required for accessing the Google Cloud Natural Language API and should be securely stored and managed.
    """

**hide**
 
    Replaces occurrences of strings in hide_list with a block character, effectively censoring them.

    Parameters:
        hide_list (list): A list of strings that need to be censored in the input text.
        leading_space_required (bool): A flag indicating whether a leading space is required before each match for it to be censored. Defaults to True.

    Returns:
        count (int): The number of substitutions made.

    This method iterates over each string in hide_list and replaces it with a block character (U+2588) in the partially hidden string. If leading_space_required is True, it ensures that each string is preceded by a space or starts at the beginning of the line for it to be censored. This approach helps in avoiding partial matches within words.

    Exceptions are caught and handled by attempting a more direct search-and-replace if the regex approach fails. This is a fallback mechanism to ensure all instances are covered, even when regex might not apply.

    Note: The method updates the partially_hidden_string attribute with the censored version of the input string.

**hide_address**

    Calls method in entity_resolver to identify physical address. Once address are complied calls hide method to hide address.
    Returns:
        int: The number of addresses censored in the input text.

**hide_names**

    Calls method in entity_resolver to identify names. Once names are complied calls hide method to hide address.
    Returns:
        int: The number of names censored in the input text.

**hide_phone_numbers**

    Calls method in entity_resolver to identify phone numbers. Once phone numbers are complied calls hide method to hide address.
    Returns:
        int: The number of phone numbers censored in the input text.

**hide_date**
 
    Calls method in entity_resolver to identify date. Once date are complied calls hide method to hide address.
    Returns:
        int: The number of date censored in the input text.

**assignment1/entityresolver.py**

**find_all_address**

    Combines results from multiple sources to find all possible addresses in the input text.

    Parameters:
        input_string (str): The original text from which addresses need to be extracted.
        doc_obj (Spacy.Doc): A Spacy Doc object of the input text, used for leveraging Spacy's NLP capabilities.
        gcp_nlp_entities (list): A list of entities identified by Google Cloud Platform's Natural Language API.

    Returns:
        list: A deduplicated list of addresses found using both PyAP for pattern-based address recognition,
              Spacy for location entities, and GCP NLP for additional location entity recognition.


**find_location_using_spacy**
    
    Extracts location entities from a text using Spacy's named entity recognition.

    Parameters:
        doc_obj (Spacy.Doc): A Spacy Doc object processed by Spacy's NLP pipeline, containing the analyzed text.

    Returns:
        list: A list of strings, each representing a location entity found in the text. This includes
              both LOC (locations like mountains, rivers) and GPE (geo-political entities like countries, cities) types.

    This function iterates over the named entities recognized by Spacy in the given document object. It filters these
    entities to include only those classified as LOC or GPE, which are indicative of locations and geo-political entities,
    respectively. The text of each qualifying entity is then appended to a list, which is returned as the function's output.

**find_location_using_gc_nlp**
    
    Extracts location entities from text using Google Cloud Natural Language Processing (NLP) API results.

    Parameters:
        gcp_nlp_entities (list): A list of entities recognized by the Google Cloud NLP API from the input text.
                                 Each entity in the list is a dictionary containing details about the recognized entity.

    Returns:
        list: A list of strings, each representing a location entity identified by the Google Cloud NLP API. 
              Only entities classified as "LOCATION" are included.

    This function iterates over the entities recognized by the Google Cloud NLP API, filtering for those
    specifically classified as "LOCATION".

**find_address_using_pyap**

    Extracts physical addresses from a given text string using the PyAP library.

    Parameters:
        input_string (str): The text string from which physical addresses are to be extracted.

    Returns:
        list: A list of strings, where each string is a full physical address identified within the input text.
              This extraction is based on the patterns recognized by PyAP specific to the country provided, in this case, 'US'.

**find_all_names**

    Combines the extraction of names from a given text string using both spaCy and Google Cloud NLP entities.

    Parameters:
        input_string (str): The text string from which names are to be extracted. This parameter is not directly used in the function but is included for consistency and potential future use.
        doc_obj (Doc): The spaCy document object obtained by processing the input string with a spaCy NLP model.
        gcp_nlp_entities (list): A list of entities recognized by the Google Cloud NLP API in the given text string.

    Returns:
        list: A deduplicated list of names identified in the input string, combining results from both spaCy and Google Cloud NLP entities.

**find_names_using_spacy**

    Extracts names and organizational entities from text using spaCy's named entity recognition.

    Parameters:
        doc_obj (spaCy.Doc): A spaCy document object that has been processed by a spaCy NLP model, containing the analyzed text.

    Returns:
        list: A list of strings, each representing either a person's name or an organization's name found in the text. This includes entities classified as "PERSON" or "ORG" by spaCy.

**find_name_using_gcp_nlp**

    Extracts names identified as personal names from the entities recognized by Google Cloud's Natural Language Processing (NLP) API.

    Parameters:
        gcp_nlp_entities (list): A list of entities identified by Google Cloud's NLP API, where each entity is represented as a dictionary containing details about the entity, including its type and name.

    Returns:
        list: A list of strings, each representing a personal name found in the text analyzed by Google Cloud's NLP API. Only entities classified as "PERSON" are included.

**find_all_date**

    Aggregates date entities from a text string using both spaCy and Google Cloud NLP.

    Parameters:
        input_string (str): The text string from which date entities are to be extracted. While this parameter is part of the method's signature for consistency and future extension, it is not directly used in the function.
        doc_obj (spaCy.Doc): A document object processed by spaCy's NLP model, used for identifying date entities within the text.
        gcp_nlp_entities (list): Entities recognized by Google Cloud's NLP API, which are analyzed to identify date entities.

    Returns:
        list: A deduplicated list of date entities found in the input text, combining the findings from both spaCy's local analysis and Google Cloud's NLP entity recognition.

**find_date_using_spacy**

    Identifies date entities within text using spaCy's named entity recognition.

    Parameters:
        doc_obj (spaCy.Doc): The spaCy document object that has been processed by spaCy's NLP pipeline. This object contains the analyzed text along with its linguistic annotations.

    Returns:
        list: A unique list of date entities recognized in the text. These dates are identified based on spaCy's entity recognition capabilities and are specifically tagged as "DATE".

**find_date_using_gcp_nlp**
    
    Extracts date entities from text analyzed by Google Cloud Natural Language API.

    Parameters:
        gcp_nlp_entities (list): A list of entities recognized by Google Cloud Natural Language API. Each entity in the list is a dictionary containing details about the recognized entity, such as its type and textual representation.

    Returns:
        list: A list of date entities extracted from the input text. These dates are identified based on the types assigned by Google Cloud Natural Language API, specifically those categorized as "DATE".

**find_all_phone_numbers**

    Aggregates potential phone numbers from a text string using a combination of methods, including a phone number matcher, Google Cloud NLP, and regex patterns.

    Parameters:
        input_string (str): The text string from which phone numbers are to be extracted. This parameter is directly utilized in regex and phone number matcher methods.
        doc_obj (spaCy.Doc): A document object processed by spaCy's NLP model. This parameter is included for consistency and potential future use but is not directly utilized in the current implementation.
        gcp_nlp_entities (list): A list of entities identified by Google Cloud's NLP API, which are analyzed to identify phone number entities.

    Returns:
        list: A deduplicated list of phone numbers found in the input text, combining findings from a phone number matching library, regex pattern matching, and Google Cloud's NLP entity recognition.

**find_phone_numbers_using_phone_number_matcher**

    Extracts phone numbers from the input string using the PhoneNumberMatcher library.

    Parameters:
        input_string (str): The text string from which phone numbers are to be extracted.

    Returns:
        list: A list of phone numbers extracted from the input string. Each phone number is represented as it appears in the text without any formatting applied.

**find_phone_numbers_using_regex**
  
    Extracts phone numbers from the input string using regular expressions.

    Parameters:
        input_string (str): The text string from which phone numbers are to be extracted.

    Returns:
        list: A list of phone numbers extracted from the input string, formatted according to the recognized patterns.

**find_phone_numbers_using_gcp_nlp**
    
    Extracts phone numbers identified by Google Cloud Natural Language Processing (NLP) API.

    Parameters:
        gcp_nlp_entities (list): A list of entities recognized by Google Cloud's NLP API, with each entity represented as a dictionary containing the entity's type and name.

    Returns:
        list: A list of phone numbers extracted from the input text.

**find_names_in_email**

    Extracts names from email addresses in the input text of a MaskString object.

    Parameters:
        maskstring_obj (MaskString): An instance of the MaskString class which contains the text to be analyzed and the necessary NLP tools and methods.

    Returns:
        list: A list of unique names extracted from email addresses found in the input text.


# tests/test_censored_file_output.py

**test_censored_file_output**

    Tests the functionality of the `create_censored_file` function to ensure it creates a censored file with the correct content.

    This test function performs the following steps:
    1. Constructs the file path and folder path for the test.
    2. Defines the expected content of the censored file.
    3. Calls `create_censored_file` to create a `.censored` file in the specified folder with the expected content.
    4. Verifies that the file is created and that its content matches the expected censored content.

    Assertions:
    - Asserts that the `.censored` file exists in the specified folder.
    - Asserts that the content of the `.censored` file matches the expected content.

# tests/test_entity_recognition.py

**mock_doc_with_entities**
 
    Creates a mock spaCy Doc object populated with predefined entities for testing purposes.

    This function sets up a mock environment for entity recognition tests by simulating a spaCy Doc object with a specific set of named entities. It is designed to support unit testing of functions that rely on spaCy's natural language processing capabilities without the need for actual NLP model processing.

    Returns:
        A mock spaCy Doc object with predefined named entities including persons, dates, geographical places, and phone numbers.

**test_name_recognition**
    
    Test function for verifying the name recognition capability.

    This function tests the ability of the `find_all_names` function to correctly identify and extract names from a given input string, using both spaCy's NLP model and Google's Cloud Natural Language Processing API results. It asserts that the names identified match the expected list of names, ensuring the accuracy of the name extraction process.
    
    The test uses an assertion to verify that the output of the `find_all_names` function matches the expected list of names, demonstrating that the function can accurately recognize and extract names from the input text. It's a crucial part of validating the effectiveness of the text processing pipeline, especially in contexts where accurate name recognition is essential, such as data redaction, information retrieval, and content personalization.

**test_address_recognition**

    Test function for verifying the address recognition capability.

    This function assesses the `find_all_address` function's ability to accurately identify and extract addresses from a provided input string. It utilizes both a mocked spaCy Doc object and simulated Google Cloud NLP API results to simulate the process of address recognition. The test ensures that the extracted addresses match an expected set of addresses, demonstrating the function's accuracy in identifying various forms of addresses within text.

    The assertion in this test checks that the list of addresses returned by the `find_all_address` function, when sorted, matches the sorted list of expected addresses. This comparison validates the comprehensive and accurate extraction of address information from the text, crucial for applications requiring precise location data extraction, such as geographic tagging, content filtering, and sensitive information redaction.

**test_phone_number_recognition**
    
    Test function for verifying phone number recognition accuracy.

    Evaluates the `find_all_phone_numbers` function to ensure it precisely identifies and extracts phone numbers from a given input string. This test leverages a combination of resources: a mocked spaCy Doc object and simulated Google Cloud NLP API entities, to assess the functionality of phone number extraction across different methods. It asserts that the list of phone numbers identified by the function matches the expected output, confirming the effectiveness of the phone number recognition process.


    The test includes an assertion to verify that the output of the `find_all_phone_numbers` function is exactly as expected, demonstrating the function's capability to accurately recognize and extract phone numbers from textual content. This is vital for applications requiring contact information processing, such as communication data analysis, privacy protection, and information redaction.

**test_date_recognition**

    Test function for verifying date recognition accuracy.

    Utilizes the `find_all_date` function to validate the precision of date extraction from a provided input string. This test employs a combination of resources: a mocked spaCy Doc object and simulated Google Cloud NLP API entities, to evaluate the functionality of date identification across different methods. It asserts that the list of dates identified by the function matches the expected output, confirming the effectiveness of the date recognition process.

    The test includes an assertion to ensure that the output of the `find_all_date` function precisely matches the expected list of dates. This verification is crucial for applications reliant on date information processing, such as scheduling systems, event management tools, and data analysis platforms.

# tests/test_mask_string.py

**test_maskstring_with_mocked_spacy_and_gcp_nlp**
    
    Test function for verifying MaskString object creation with mocked spaCy and Google Cloud NLP.

    This test validates the behavior of the MaskString class when using mocked spaCy and Google Cloud NLP functionalities. It utilizes the @patch decorator from the pytest library to substitute the spacy.load and MaskString.gcp_nlp methods with mock objects, allowing controlled simulation of NLP processing. The test initializes a MaskString object, sets an input string, and verifies that both mocked methods are called appropriately.

    Parameters:
        mock_gcp_nlp (MagicMock): Mock object for the MaskString.gcp_nlp method, returning predefined entities.
        mock_spacy_load (MagicMock): Mock object for the spacy.load method, returning a mocked spaCy NLP model.

    This test ensures that MaskString functions as expected even in scenarios where actual NLP processing is replaced with mocks, demonstrating its robustness and compatibility with external NLP services.

# tests/test_stats_output.py

**test_stat_file_output**
    
    Test function for verifying the output of statistics to a file.

    This test validates the functionality of writing statistics to a file using the write_stats function from the censoror module. It creates a temporary file path, generates sample statistics, calls the write_stats function to write the statistics to the file, and then verifies that the file was created and its contents match the expected statistics. 
