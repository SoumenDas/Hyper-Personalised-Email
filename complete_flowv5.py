import gradio as gr
import json
import base64
from PIL import Image
import openai
import requests

import os
from dotenv import load_dotenv

import json

import os

import base64
import json
from typing import List, Dict
from PIL import Image
import requests
from openai import OpenAI
import sendgrid
import os
from sendgrid.helpers.mail import Mail, Email, To, Content

load_dotenv(override=True)

# Configure your API keys here
OPENAI_API_KEY = os.getenv("OPENAI_API_KEY")

# Hardcoded URLs
USER_EVENTS_URL = "https://ik.imagekit.io/dt2cpdsmtz/user_events.json"  # Replace with actual URL
INVENTORY_URL = "https://ik.imagekit.io/dt2cpdsmtz/inventory.json"      # Replace with actual URL
USER_PHOTO_URL = "https://ik.imagekit.io/dt2cpdsmtz/model_data/model_2.jpg"  # Replace with actual URL


class PersonalizedMarketingApp:
    def __init__(self):
        self.openai_client = openai.OpenAI(api_key=OPENAI_API_KEY)
        self.persona_data = None
        self.analysis = None
        self.generated_images = []
        self.email_html = ""
        self.user_photo_url = USER_PHOTO_URL
    
    def analyze_user_events(self, progress=gr.Progress()):
        """Step 1: Analyze user events and infer persona, timing, and channel preferences"""
        progress(0, desc="Starting analysis...")
        
        try:
            progress(0.2, desc="Fetching user events data...")
            
            # Fetch the user events JSON from URL
            try:
                response = requests.get(USER_EVENTS_URL, timeout=30)
                response.raise_for_status()  # Raise an exception for bad status codes
                events_data = response.text
            except requests.RequestException as e:
                # Fallback to sample data if URL fetch fails
                print(f"Failed to fetch user events from URL: {e}")
                events_data = """{
                    "user_id": "sample_user_123",
                    "events": [
                        {"event": "product_view", "category": "fashion", "time": "2024-01-15T10:30:00"},
                        {"event": "add_to_cart", "category": "casual_wear", "time": "2024-01-15T10:35:00"}
                    ]
                }"""
            
            progress(0.4, desc="Preparing analysis prompt...")
            
            # Prepare prompt for LLM analysis
            prompt = f"""You are analyzing user behavior from fashion ecommerce app events. Based on the user events data provided, create a detailed persona analysis.

            Analyze the user events and provide a JSON response with the following structure:
            {{
                "userId": "extracted user ID from events",
                "journey": "detailed user journey description",
                "characteristics": "behavioral characteristics and patterns",
                "persona_description": "detailed persona description",
                "best_day": "recommended day for communications",
                "best_time": "recommended time for communications", 
                "best_channel": "recommended communication channel",
                "reasons": "detailed reasoning for timing and channel recommendations"
            }}

            User Events Data:
            {events_data}

            Provide only the JSON response without any additional formatting or markdown."""

            progress(0.6, desc="Calling OpenAI API...")
            
            # Use OpenAI directly instead of the agents library
            response = self.openai_client.chat.completions.create(
                model="gpt-4o-mini",
                messages=[
                    {"role": "system", "content": "You are a user behavior analyst specializing in ecommerce customer personas. Always respond with valid JSON only."},
                    {"role": "user", "content": prompt}
                ],
                max_tokens=1500,
                temperature=0.7
            )
            
            progress(0.8, desc="Processing response...")
            
            # Parse the response
            response_content = response.choices[0].message.content.strip()
            
            # Remove markdown code blocks if present
            if response_content.startswith("```json"):
                response_content = response_content[7:]
            if response_content.startswith("```"):
                response_content = response_content[3:]
            if response_content.endswith("```"):
                response_content = response_content[:-3]
            
            # Parse JSON response
            import json
            self.persona_data = json.loads(response_content.strip())
            print("Persona data extracted:", self.persona_data)
            
            progress(1.0, desc="Analysis complete!")
            
            # Format for display
            display_text = f"""
                ## 📊 User Analysis Results

                ### 👤 Persona Profile
                - **UserID**: {self.persona_data.get('userId', 'N/A')}
                - **Journey**: {self.persona_data.get('journey', 'N/A')}
                - **Behavior Patterns**: {self.persona_data.get('characteristics', 'N/A')}
                - **Persona**: {self.persona_data.get('persona_description', 'N/A')}

                ### ⏰ Optimal Communication Strategy
                - **Best Time**: {self.persona_data.get('best_time', 'N/A')}
                - **Best Day**: {self.persona_data.get('best_day', 'N/A')}
                - **Best Channel**: {self.persona_data.get('best_channel', 'N/A')}
                - **Reasons**: {self.persona_data.get('reasons', 'N/A')}
                """
            
            return display_text, gr.update(visible=True), USER_PHOTO_URL
            
        except json.JSONDecodeError as e:
            return f"Error parsing AI response as JSON: {str(e)}\nResponse was: {response_content[:200]}...", gr.update(visible=False), None
        except Exception as e:
            return f"Error analyzing user events: {str(e)}", gr.update(visible=False), None
        
    def generate_recommendations(self, progress=gr.Progress()):
        """Step 2: Generate product recommendations based on user photo URL and persona"""
        progress(0, desc="Starting recommendation generation...")
        
        # Initialize the recommender
        recommender = ImageProductRecommender()
        
        progress(0.3, desc="Analyzing image and generating recommendations...")
        
        # Analyze the image
        print("🔍 Analyzing image and generating recommendations...")
        recommendations = recommender.analyze_image(self.persona_data, self.user_photo_url)
        print("Recommendations generated:", recommendations)
        
        try:
            progress(0.7, desc="Processing recommendations...")
            
            self.analysis = recommendations
            
            progress(1.0, desc="Recommendations complete!")
            
            # Format for display
            display_text = "## 🛍️ Personalized Product Recommendations\n\n"
            for i, rec in enumerate(recommendations['recommendations'], 1):
                display_text += f"""
                ### {i}. {rec['product_name']}
                - **Price**: {rec['price_range']}
                - **Why it fits**: {rec['reason']}
                - **Use case**: {rec['use_case']}
                - **Style Category**: {rec['category']}
                ---
                """
            
            return display_text, gr.update(visible=True)
            
        except Exception as e:
            return f"Error generating recommendations: {str(e)}", gr.update(visible=False)
    
    def send_personalized_email(self, recipient_email, progress=gr.Progress()):
        """Combined Step: Generate images, create email, and send it"""
        progress(0, desc="Starting personalized email process...")
        
        try:
            # Step 1: Generate product images
            progress(0.2, desc="Generating product images...")
            
            generated_images = []
            model_path = self.user_photo_url
            
            def _create_image(_model_path, _garment_image_path):
                """Helper function to create image using Fashn API"""
                
                API_KEY = os.getenv("FASHN_API_KEY")
                assert API_KEY, "Please set the FASHN_API_KEY environment variable."
                BASE_URL = "https://api.fashn.ai/v1"
                
                input_data = {
                    "model_image": _model_path,
                    "garment_image": _garment_image_path,
                    "category": "auto",
                }
                headers = {"Content-Type": "application/json", "Authorization": f"Bearer {API_KEY}"}
                run_response = requests.post(f"{BASE_URL}/run", json=input_data, headers=headers)
                run_data = run_response.json()
                prediction_id = run_data.get("id")
                
                while True:
                    status_response = requests.get(f"{BASE_URL}/status/{prediction_id}", headers=headers)
                    status_data = status_response.json()
                
                    if status_data["status"] == "completed":
                        return status_data["output"][0]
            
            total_recs = len(self.analysis.get('recommendations', []))
            
            for i, rec in enumerate(self.analysis.get('recommendations', []), 1):
                progress(0.2 + (i/total_recs * 0.3), desc=f"Generating image {i}/{total_recs}...")
                
                garment_image_path = f"https://ik.imagekit.io/dt2cpdsmtz/cloth_data/{rec.get('product_id', 'N/A')}.jpg"
                rec['product_url'] = _create_image(model_path, garment_image_path)        
                generated_images.append(rec['product_url'])
        
            self.generated_images = generated_images
            
            # Step 2: Generate email content
            progress(0.6, desc="Generating email content...")
            
            # Initialize generator
            API_KEY = os.getenv('OPENAI_API_KEY')
            generator = FashionEmailGenerator(API_KEY)
            
            # Example usage with predefined persona
            persona = self.persona_data['persona_description']
    
            # Generate main fashion email
            email_html = generator.generate_fashion_email(self.analysis, persona, self.generated_images, "new_professional_collection")
            
            # Clean up the HTML (remove markdown code blocks if present)
            if email_html.startswith("```html"):
                email_html = email_html[7:]
            if email_html.endswith("```"):
                email_html = email_html[:-3]
            
            self.email_html = email_html
            
            # Step 3: Send email
            progress(0.8, desc="Sending email...")
            
            sg = sendgrid.SendGridAPIClient(api_key=os.environ.get('SENDGRID_API_KEY'))
            from_email = Email("soumenfiction@gmail.com")  # Change to your verified sender
            to_email = To(recipient_email)
            content = Content("text/html", self.email_html)
            subject = "You would look lovely in these new collections!"
            
            mail = Mail(from_email, to_email, subject, content).get()
            response = sg.client.mail.send.post(request_body=mail)
            
            progress(1.0, desc="Email sent successfully!")
            
            return f"Done 👍"
            
        except Exception as e:
            return f"❌ Error in personalized email process: {str(e)}", ""


class ImageProductRecommender:
    def __init__(self, api_key: str = None):
        """Initialize the recommender with OpenAI API key."""
        self.api_key = os.getenv('OPENAI_API_KEY')
        if not self.api_key:
            raise ValueError("OpenAI API key is required. Set OPENAI_API_KEY environment variable.")
        
        self.client = OpenAI(api_key=self.api_key)
    
    def analyze_image(self, persona, image_path: str) -> Dict:
        """Analyze image and generate product recommendations."""
        print(f"Analyzing image: {image_path}")
        
        # Create the prompt for product recommendations
        prompt = f"""
        Analyze this image carefully and provide product recommendations based on what you see. Consider user's persona in the recommendation. This is the user's persona: {persona['persona_description']}. You also need to provide recommendations only within this set of products:
        ==================================================
        Inventory Data:
        ==================================================
        ID: 00071_00 | Product: Reebok Tshirt Black | Category: Athletic Wear
        ID: 00069_00 | Product: Floral Top | Category: Traditional Wear
        ID: 00067_00 | Product: Blue Athletic Top | Category: Athletic Wear
        ID: 00064_00 | Product: Full sleaves Floral Top | Category: Casual Wear
        ID: 00057_00 | Product: Black Lee Tshirt | Category: Casual Wear
        ID: 00055_00 | Product: Purple Vans Tshirt | Category: Casual Wear
        ID: 00035_00 | Product: Long Sleeves Navy Blue Tshirt | Category: Casual Wear
        ID: 00034_00 | Product: Sleeveless Pink Top | Category: Casual Wear
        ID: 00017_00 | Product: Red Casual Tshirt | Category: Casual Wear
        ID: 00013_00 | Product: Pink Casual Tshirt | Category: Casual Wear
        ID: 00008_00 | Product: Red Full Sleeves Running Tshirt | Category: Athletic Wear
        ID: 00006_00 | Product: Long Sleeve Casual Top | Category: Casual Wear
        

        Please provide your response in the following JSON format:
        {{
            "image_description": "Brief description of what's in the image",
            "detected_items": ["list", "of", "items", "or", "products", "visible"],
            "style_category": "category like fashion, home decor, food, technology, etc.",
            "recommendations": [
                {{
                    "product_name": "Product Name",
                    "category": "Product Category",
                    "reason": "Why this product complements or relates to the image",
                    "use_case": "How the user would use this product",
                    "price_range": "Estimated price range",
                    "product_id": "Id of the product"
                }}
            ]
        }}

        Focus on:
        1. Complementary products that would pair well with what's shown
        2. Similar style products in the same category
        3. Accessories or add-ons that enhance the main items
        4. Alternative products that serve similar purposes
        
        Provide 3-5 specific, realistic product recommendations.
        """
        
        try:
            message = {
                "role": "user",
                "content": [
                    {"type": "text", "text": prompt},
                    {
                        "type": "image_url",
                        "image_url": {
                            "url": image_path,
                            "detail": "high"
                        }
                    }
                ]
            }

            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    message
                ],
                max_tokens=2000
            )
            
            # Extract and parse the response
            content = response.choices[0].message.content
            
            # Try to parse JSON from the response
            try:
                # Find JSON content in the response
                start_idx = content.find('{')
                end_idx = content.rfind('}') + 1
                json_str = content[start_idx:end_idx]
                result = json.loads(json_str)
                return result
            except json.JSONDecodeError:
                # If JSON parsing fails, return a structured response anyway
                return {
                    "image_description": "Analysis completed",
                    "detected_items": [],
                    "style_category": "general",
                    "recommendations": [],
                    "marketing_copy": content,
                    "raw_response": content
                }
                
        except Exception as e:
            raise Exception(f"Error calling OpenAI API: {str(e)}")
    
class FashionEmailGenerator:
    def __init__(self, api_key: str):
        """
        Initialize the fashion marketing email generator with OpenAI API key
        """
        self.client = openai.OpenAI(api_key=api_key)
    
    
    def create_fashion_marketing_prompt(self, persona: str, analysis, campaign_type: str = "new_collection") -> str:
        """
        Create a professional fashion marketing prompt that emphasizes customer value and personalization
        """
        recom_txt = '\n'.join(f"{rec['product_name']} - {rec['product_url']}" for rec in analysis.get('recommendations', []))
        prompt = f"""
            Create a professional fashion newsletter email for an online clothing store. 
            
            CUSTOMER STYLE PROFILE:
            {persona}
            
            FEATURED PRODUCTS:
            {recom_txt}
            
            TASK: Create a fashion newsletter email that:
            1. Welcomes the customer warmly
            2. Showcases the featured clothing items
            3. Explains the style benefits of each piece
            4. Provides outfit coordination tips
            5. Includes seasonal fashion advice
            6. Has a friendly, helpful tone
            
            EMAIL STRUCTURE:
            - Subject line about new fashion arrivals
            - Warm greeting to valued customer
            - Brief introduction about new collection
            - Feature each product with styling tips
            - Include the product image URLs provided
            - Professional closing with store information
            - Use clean HTML formatting with inline CSS
            - Make it mobile-friendly
            
            Focus on helping customers make informed fashion choices and building a positive shopping experience. Reply with the email content in HTML format only, without any markdown or code blocks.
            """
        return prompt
    
    def generate_fashion_email(self, analysis, persona: str, image_urls: List[str], campaign_type: str = "new_collection") -> str:
        """ Generate personalized fashion marketing email"""
        try:
            # Create message content
            message_content = [
                {
                    "type": "text",
                    "text": self.create_fashion_marketing_prompt(persona, analysis, campaign_type)
                }
            ]
            
            # Add fashion images using URLs directly
            for image_url in image_urls:
                message_content.append({
                    "type": "image_url",
                    "image_url": {
                        "url": image_url,
                        "detail": "high"
                    }
                })
            
            # Make API call with fashion-specific system message
            response = self.client.chat.completions.create(
                model="gpt-4o",
                messages=[
                    {
                        "role": "system",
                        "content": """You are a professional fashion marketing consultant specializing in personalized styling advice and customer-focused email campaigns. You help fashion ecommerce businesses create authentic, valuable communications that assist customers in making informed style choices. Your expertise includes fashion trends, styling techniques, fabric knowledge, and personalized recommendations that genuinely benefit the customer's wardrobe and lifestyle."""
                    },
                    {
                        "role": "user",
                        "content": message_content
                    }
                ],
                max_tokens=1200,
                temperature=0.7
            )
            
            return response.choices[0].message.content
            
        except Exception as e:
            return f"Error generating fashion email: {str(e)}"
        

# Initialize the app
app = PersonalizedMarketingApp()

# Create Gradio interface
def create_interface():
    with gr.Blocks(title="Walmart Personalized Marketing Platform", theme=gr.themes.Soft()) as demo:
        gr.Markdown("# 🚀 Walmart Personalized Marketing Platform")
        gr.Markdown("Analyze user personas, generate recommendations, and create personalized emails!")
        
        with gr.Row():
            with gr.Column(scale=1):
                # Step 1: Analyze User Events
                gr.Markdown("## Step 1: Analyze User Data")
                gr.Markdown("*Using user events data from URL*")
                analyze_btn = gr.Button("Analyze User Data", variant="primary")
                
                # Analysis Results
                analysis_output = gr.Markdown(visible=True, label="Analysis Results")
                
                with gr.Group(visible=False) as step2_group:
                    gr.Markdown("## Step 2: User Photo & Recommendations")
                    gr.Markdown("*User photo for personalization:*")
                    user_photo_display = gr.Image(
                        label="User Photo",
                        type="filepath",
                        interactive=False,
                        height=300
                    )
                    generate_recs_btn = gr.Button("Generate Recommendations", variant="primary")
                
                recommendations_output = gr.Markdown(visible=True, label="Recommendations Results")
                
            with gr.Column(scale=1):
                with gr.Group(visible=False) as email_group:
                    gr.Markdown("## Step 3: Send Personalized Email")
                    gr.Markdown("*This will generate product images, create email content, and send the email*")
                    recipient_email = gr.Textbox(
                        label="Recipient Email",
                        placeholder="Enter email address"
                    )
                    send_personalized_email_btn = gr.Button("Send Personalised Email", variant="primary", size="lg")
                    send_status = gr.Markdown()
                    
                    
        
        # Event handlers with progress tracking
        analyze_btn.click(
            fn=app.analyze_user_events,
            inputs=[],
            outputs=[analysis_output, step2_group, user_photo_display],
            show_progress=True
        )
        
        generate_recs_btn.click(
            fn=app.generate_recommendations,
            inputs=[],
            outputs=[recommendations_output, email_group],
            show_progress=True
        )
        
        send_personalized_email_btn.click(
            fn=app.send_personalized_email,
            inputs=[recipient_email],
            outputs=[send_status],
            show_progress=True
        )
    
    return demo

# Launch the app
if __name__ == "__main__":
    demo = create_interface()
    demo.launch(
        server_name="0.0.0.0",
        server_port=7860,
        share=True,
        debug=True
    )