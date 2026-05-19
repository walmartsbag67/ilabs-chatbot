# 3D Printer
1. Safety First: 
Moving Parts: Keep your hands clear of the printer's interior during operation as the build plate moves with enough force to cause injury.
. 
Cooldown: Always allow the printer to cool for at least 5 minutes before reaching inside.

2. Compatible File Formats
To begin, your 3D model must be saved or exported in a format that Ultimaker Cura (the preparation software) can recognize
. Supported file types include:
STL, OBJ, 3MF, and X3D
.
3. Preparing the Print (Ultimaker Cura)
Ultimaker Cura is used to "slice" your 3D model, converting it into instructions (GCODE) that the printer can follow
.
Load the Model: Open Cura and click the “Open File” folder icon to import your design
.
Check Configuration: Verify in the right-hand sidebar that the print cores (e.g., AA 0.4) and materials (e.g., PLA) match what is physically loaded in the printer
.
Adjust Your Design: Use the Adjustment Tools on the left to move, scale, or rotate your model on the virtual build plate
.
You can print multiple projects at once if there is enough space by dragging other files to the previous tab.
.
To turn on support structures in UltiMaker Cura, simply open the print settings panel, switch to the Custom view, and check the Generate Support box.
1.Select your settings view: On the right side of the screen, click on the Print Settings panel (usually labeled with your current profile, like Standard Quality).
2.Switch to custom settings: Select the Custom tab to unlock advanced options.
3.Turn on supports: Scroll down to the Support section and check the box that says Generate Support.
.
ConfigSetup: Undeure Print r the “Recommended” or “Custom” tabs, choose your layer height (detail vs. speed) and infill (density)
.
Slicing and Preview: Once settings are chosen, the software slices the model. Change the view mode to “Layer view” to inspect the print path for every individual layer to catch errors before printing
.
4. Transferring via USB Drive
If you are not printing over a network, you must transfer the sliced file using a USB stick
:
Save the File: Insert your USB stick into the computer. In Cura, click “Save to removable drive” or “Save to USB” to export the .gcode file
.
Eject Safely: Always eject the USB stick in the software before removing it from your computer to avoid file corruption
.
Insert into Printer: Plug the USB drive into the port on the front of the Ultimaker 3
.
5. Starting and Removing the Print
Initiate: Navigate the printer’s display menu to Print, select your file from the USB list, and push the button to confirm
.
Removal: Once finished, allow the build plate to cool down; the material will contract as it cools, making it easier to pop the print off
.
Cleanup: After removal, clean the glass plate with a scraping tool
.
6. Common Troubleshooting
Warping: If the corners of your print lift, ensure the build plate is level and that you have applied enough glue
.
Under-extrusion: If you see missing layers or random holes, it may be due to a partial clog in the print core or incorrect feeder tension
.
Clogged Print Core: If no material flows for 10 minutes, the core is likely clogged and requires a “hot and cold pull” cleaning procedure
.
7. Overnight printing
Overprinting is only allowed on weekdays except weekends and holidays.

# LASER CUTTER

1. Preparation and File Handling
Software: Designs must be prepared in the designated controlling software (as detailed in the separate Software Operating Manual) and transmitted to the machine's MPC6515 controller
.
File Selection: Files are saved by name on the PAD03 operation panel. You can select your file using the arrow keys
.
Materials: This machine is strictly for nonmetal materials such as acrylic (Lucite), wood, rubber, and plastic
.
2. Safety and Startup
Water Cooling (Mandatory): Always start the submersible pump and blower first. Ensure water is circulating and below 35°C before powering on the laser
.
Reset: Upon power-up, the laser head will automatically move to its initial state (usually the upper right corner) once it receives the reset signals
.
General Rules: Never leave the machine unattended while it is working, and do not place your hands in the workplace during operation
.
3. Setting Your Starting Point
Manual Positioning: Clear any active menu selections by pressing "Esc". Use the direction (arrow) keys to move the laser head to where you want your job to begin
.
Focusing: Place your material and adjust the height. For the standard 50mm lens, the gap between the material surface and the laser head's undersurface must be 20mm
.
Trace/Verify:
Test Key: Press this to have the head run along the outline border of your data without firing the laser
.
Boundary Cut (CUT BDR): Use this menu option to cut a rectangular boundary around your design to verify the exact location on the material
.
4. Engraving and Cutting (Power & Speed)
The machine's effectiveness depends on the relationship between these two settings, which can be adjusted on the main interface or in real-time during a job
.
Engraving: Generally requires higher speeds and lower power to create surface patterns
.
Cutting: Generally requires lower speeds and higher power to penetrate the material
.
Current Limit: To protect the laser tube, keep the electric current under 20mA (ideally 13–17mA)
.
Adjusting Mid-Job: Use the up/down arrows to change power and left/right arrows to change speed while the machine is running
.
5. Maintenance and Troubleshooting
Maintenance: Regularly clean the lenses and reflectors with absolute alcohol and cotton
. Lubricate the linear bearings with sewing machine oil before and after use
.
Rest Cycles: After three consecutive hours of work, the machine must be shut down for 30 minutes to cool
.
Troubleshooting Depth: If the cut or engraving is too shallow, check for polluted lenses, increase power, or decrease the processing speed
.

# Sunway

Sunway iLabs is a comprehensive innovation ecosystem designed for entrepreneurs at all stages, serving as the innovation lab for both the Sunway Group and Sunway University
Sunway iLabs is a non-profit innovation ecosystem and incubator built through a partnership between Sunway Group and Sunway University. It is designed to foster entrepreneurship, help students build minimum viable products (MVPs), and provide access to technical tools in the Makerspace.
. Its core mission is centered on a three-part framework: Inspire, Build, and Scale
.
Core Pillars & Mission
Inspire (Education): Acting as a catalyst for innovation and growth, this pillar provides entrepreneurship education, hackathons, and a university startup accelerator program specifically for the next generation of entrepreneurs
.
Build (Venture Labs): This pillar focuses on creating robust, sustainable solutions. It transforms AI and DeepTech ideas into scalable startups and assists global companies in expanding into Malaysia and Southeast Asia by providing talent access and strategic investment partnerships
.
Scale (Venture Capital): This phase focuses on long-term business success and impact. It involves investing in startups and venture capital funds that create synergies with the Sunway Group, aiming to empower innovation in Malaysia and beyond
.
Specialized Labs
The ecosystem operates two primary labs to facilitate its growth initiatives:
Startup Deep Tech Ventures Lab: Focused on high-tech innovation
.
Cross Border Market Access Lab: Focused on helping international entities enter the regional market
.
Program Offerings for Students
Specifically categorized "For Students," the Education pillar includes:
Academic Programme: Structured educational offerings for example Entrepeneurial Mindset & Skills and Paradox of Theron (PoT)

Entrepeneurial Mindset & Skills
In the context of the Sunway University and Sunway iLabs ecosystem, EMS stands for Entrepreneurial Mindset & Skills.  

Far from being a dry, textbook-heavy lecture, EMS is a compulsory, award-winning signature course managed through the Startup Foundry at Sunway iLabs. It is a mandatory requirement for all undergraduate students under Sunway's holistic S.U.S.T.A.I.N. educational framework.  

The program bridges classroom theory with real-world, hands-on startup execution using the following pillars:

1. The Core Competencies
The EMS curriculum focuses on six critical life and business competencies designed to prepare students for an ambiguous, fast-changing economy:  

Opportunity Recognition: Spotting actual market gaps or inefficiencies and framing them as solutions.  

Perseverance & Resilience: Learning how to treat failure as a strategic milestone rather than a dead end.

Critical Thinking & Problem Solving: Deconstructing complex, multifaceted layout or technical hurdles.

Creativity & Innovation: Challenging traditional patterns to build original assets.  

Collaboration & Communication: Working across different academic faculties to build multi-disciplinary teams.  

Calculated Risk-Taking: Making high-stakes executive decisions backed by initial data validation.

2. Immersive Game-Based Learning: Paradox of Theron (POT)
Instead of sitting through standard exams, EMS utilizes an innovative, in-house online serious game called The Paradox of Theron.  

The Sandbox: Over a multi-week simulation (typically 21 days), thousands of students manage virtual companies simultaneously.  

The Challenge: You are forced to make immediate decisions on supply chains, asset allocation, and resource boundaries, all while keeping your venture aligned with the United Nations Sustainable Development Goals (SDGs).

3. The iLabs Materialization: Slicing into Reality
The EMS framework is directly synchronized with the physical Sunway iLabs Makerspace and its broader incubation programs (like LaunchX or the Startup Bootcamp).

Students are encouraged to take the intellectual concepts tested inside the EMS curriculum and use the lab to build actual, physical Minimum Viable Products (MVPs). This transitions your learning directly into the execution phase:

Technical Validation: Moving from abstract business ideas to physical prototyping using the lab's Ultimaker 3 3D printers or running vector engraving framing on the Sunway 5030 CO2 Laser Cutter.

Digital Infrastructure: Building landing pages, connecting backend database architectures (like XAMPP/MySQL solutions), and testing interactive custom software tools.

Venture Pitching: Refining your financial models, unit economics, and go-to-market strategies to pitch to actual investors at the end-of-semester Demo Day.  

In short, EMS at Sunway isn't an isolated academic box; it is an integrated launchpad that provides the exact mental sandbox and technical infrastructure needed to turn a student project into a fully validated startup.

Paradox Of Theron
What is "The Paradox of Theron" Game?
Instead of forcing thousands of students to learn startup business theory through traditional, dry lectures, Sunway gamified the process.

The Simulation: POT is an online, serious immersive game where students navigate real-world business and sustainability challenges over a multi-week period (typically around 21 days).

The Risk-Free Sandbox: You are forced to make tough entrepreneurial decisions, test aggressive market strategies, manage limited resources, and deal with unexpected failures—all in a simulated ecosystem where mistakes won't cost real money.

The Sustainability Angle: The game directly embeds United Nations Sustainable Development Goals (SDGs). It forces you to balance making a profit with ethical governance and planetary health.

🔄 The Flipped Classroom Model
Sunway runs the Paradox of Theron alongside face-to-face facilitated sessions:

Online Gameplay: You play the game online to encounter structural roadblocks and test theories independently.

Collaborative Problem-Solving: You then bring those practical game results (like a failed strategy or supply chain block) into classroom discussions or places like iLabs to troubleshoot them using actual business frameworks.
.
Non-Academic Programme: Complementary learning opportunities
.
LaunchX: An initiative to support student ventures
.
Alumni Insider: A program to maintain a network of past participants
.
Venture Capital & Investment
Sunway iLabs functions as a strategic investor, targeting:
Startups and VC funds that offer potential for synergy with the Sunway Group
.
The goal is to foster scaling for business success while creating a significant regional impact
.
Physical Hub and Digital Presence
Physical Hub: Sunway FutureX, located at Duplex, Jalan PJS 11/26, Bandar Sunway, 47500 Subang Jaya, Selangor
.
.

# BOOKING EQUIPMENT
- **General Access:** All Sunway students must book equipment via the official iLabs website before use.
- **Equipment:** 3D Printer, Laser Cutter, Sewing Machine.
- **URL:** [(https://bookings.cloud.microsoft/book/iLabsFoundyMakerspaceFacilitiesBooking@sunway.edu.my/?ismsaljsauthenabled=true)]