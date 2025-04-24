
from flask import Flask, render_template, request, url_for, redirect, jsonify # Added jsonify
import pandas as pd
import pickle
import smtplib # For sending email
from email.mime.text import MIMEText # For creating email messages
import os # Still needed for other potential env vars, but not credentials
# --- End Imports ---

app = Flask(__name__)

# --- Load your model and data (existing code) ---
try:
    with open('model.pkl', 'rb') as f:
        model, symptoms, le = pickle.load(f)
except FileNotFoundError:
    print("Error: model.pkl not found. Please ensure the model file is in the correct directory.")
    exit()
except Exception as e:
    print(f"Error loading model.pkl: {e}")
    exit()

try:
    # Load supporting CSVs
    desc_df = pd.read_csv('symptom_Description.csv')
    precaution_df = pd.read_csv('symptom_precaution.csv')
    doctor_df = pd.read_csv('doctor_specialist.csv')
    med_df = pd.read_csv('medications.csv')
except FileNotFoundError as e:
     print(f"Error: Required CSV file not found ({e}). Please ensure all CSVs are present.")
     exit()
except Exception as e:
    print(f"Error loading CSV files: {e}")
    exit()
# --- End Load Data ---

# --- Login Route ---
@app.route('/login')
def login():
    # Renders the login page
    return render_template('login.html')

# --- Main Page Route (Symptom Selection) ---
@app.route('/')
def index():
    # Renders the main prediction interface page
    # The JS guard in index.html will handle redirection if not logged in
    return render_template('index.html', symptoms=symptoms)

# --- Prediction Route (handles form submission) ---
@app.route('/predict', methods=['POST'])
def predict():
    # This route should only be reachable after logging in and submitting the form
    # The result.html template will also have a login guard

    if request.method == 'POST':
        selected = request.form.getlist('symptoms')

        # Basic check if symptoms were selected
        if not selected:
             # Redirect back to index (required attribute should prevent this)
             return redirect(url_for('index'))

        # --- Your existing prediction logic ---
        try:
            input_data = [1 if s in selected else 0 for s in symptoms]

            prediction = model.predict([input_data])[0]
            predicted_disease = le.inverse_transform([prediction])[0]

            # Description
            description_row = desc_df[desc_df['Disease'] == predicted_disease]
            description = description_row['Description'].values[0] if not description_row.empty else "No description available."

            # Precautions
            precautions_row = precaution_df[precaution_df['Disease'] == predicted_disease]
            precautions = precautions_row.iloc[0, 1:].dropna().tolist() if not precautions_row.empty else []

            # Specialist
            specialist_row = doctor_df[doctor_df['Disease'] == predicted_disease]
            specialist = specialist_row['Specialist'].values[0] if not specialist_row.empty else "General Physician"

            # Medications
            med_row = med_df[med_df['Disease'] == predicted_disease]
            medications = med_row['Medication'].values[0].split(',') if not med_row.empty else []
            medications = [med.strip() for med in medications if med.strip()] # Clean up list

            # --- End Prediction Logic ---

            return render_template('result.html',
                                   disease=predicted_disease,
                                   description=description,
                                   precautions=precautions,
                                   specialist=specialist,
                                   selected_symptoms=selected,
                                   medications=medications)
        except Exception as e:
            print(f"Error during prediction or data lookup: {e}")
            # Render result template with an error indicator
            return render_template('result.html', error=f"An error occurred during prediction: {e}")

    # If accessed via GET, redirect to the main page
    return redirect(url_for('index'))


# --- Route for Handling Consultation Form Submission ---
@app.route('/send-consultation-email', methods=['POST'])
def handle_consultation():
    if request.method == 'POST':
        data = request.get_json()

        if not data:
            return jsonify({'success': False, 'message': 'No data received'}), 400

        # --- !!! SECURITY WARNING: Hardcoded Credentials Below !!! ---
        # Storing email credentials directly in code is highly insecure.
        sender_email = "aishwarychaurasia9@gmail.com"
        # --- Using the App Password you provided ---
        sender_password = "azsv ojun qgzc qqhw" # <-- UPDATED WITH YOUR APP PASSWORD
        # --- End Hardcoded Credentials ---

        # Default to Gmail SMTP, change if needed (can still use env vars for these)
        smtp_server = os.environ.get('MY_SMTP_SERVER', 'smtp.gmail.com')
        smtp_port = int(os.environ.get('MY_SMTP_PORT', 587)) # 587 is standard for TLS

        recipient_email = data.get('email') # Get recipient from submitted form data

        # Basic check for recipient
        if not recipient_email:
             print("Recipient email missing from form data.")
             return jsonify({'success': False, 'message': 'Recipient email address is required.'}), 400

        # --- Construct Email ---
        subject = f"Consultation Request Received: {data.get('predictedDisease', 'Health Query')}"
        body = f"""
        Dear {data.get('name', 'User')},

        Thank you for submitting a consultation request via the Personalised Health Assistant.
        We have received your details regarding a potential consultation for '{data.get('predictedDisease', 'the predicted condition')}' with a '{data.get('consultingSpecialist', 'Specialist')}'.

        Request Summary:
        --------------------
        Name: {data.get('name')}
        Email: {data.get('email')}
        Age: {data.get('age', 'N/A')}
        Gender: {data.get('gender', 'N/A')}
        Preferred Date: {data.get('preferredDate')}
        Preferred Timings: {data.get('preferredTimings', 'N/A')}
        Meeting Mode: {data.get('meetingMode')}
        Request Time: {data.get('requestTimestamp')}

        A representative or the specialist's office will attempt to contact you based on the information provided. Please note this is an automated confirmation.

        Best regards,
        The Personalised Health Assistant Team
        """

        message = MIMEText(body)
        message['Subject'] = subject
        message['From'] = sender_email
        message['To'] = recipient_email
        # You might want to add a CC or BCC to yourself or an admin email
        # message['Bcc'] = 'admin_email@example.com'

        # --- Send Email using smtplib ---
        try:
            print(f"Attempting to send email to {recipient_email} from {sender_email} via {smtp_server}:{smtp_port}")
            # Connect to server and send email
            with smtplib.SMTP(smtp_server, smtp_port) as server:
                server.ehlo()  # Can be omitted
                server.starttls()  # Secure the connection
                server.ehlo()  # Can be omitted
                server.login(sender_email, sender_password) # Use the hardcoded App Password
                server.sendmail(sender_email, recipient_email, message.as_string())
                print("Email sent successfully!")
            # Return success response to the JavaScript fetch call
            return jsonify({'success': True, 'message': 'Consultation request received and confirmation email sent.'}), 200

        except smtplib.SMTPAuthenticationError:
            # If this error still occurs, the App Password might have been revoked or entered incorrectly
            print(f"SMTP Authentication Error: Failed to authenticate with {sender_email} using the provided App Password.")
            return jsonify({'success': False, 'message': 'Server authentication failed. Please check the App Password in the code.'}), 500
        except smtplib.SMTPConnectError:
            print(f"SMTP Connection Error: Failed to connect to {smtp_server}:{smtp_port}.")
            return jsonify({'success': False, 'message': 'Server connection error. Could not send email.'}), 500
        except Exception as e:
            # Catch other potential errors (network, etc.)
            print(f"Error sending email: {e}")
            return jsonify({'success': False, 'message': f'An unexpected error occurred: {e}'}), 500
        # --- End Email Sending ---

    else:
        # Handle cases where the route is accessed via GET etc.
        return jsonify({'success': False, 'message': 'Method not allowed. Please use POST.'}), 405
# --- End Consultation Route ---


# --- Main Execution ---
if __name__ == '__main__':
    # Be aware of the security risk of hardcoding credentials above.
    app.run(debug=True) # Keep debug=True for development, set to False for production
# --- End Main Execution ---