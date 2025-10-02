# Score Vision (SN44)

Score Vision is a decentralised computer vision framework built on Bittensor that drastically reduces the cost and time required for complex video analysis. By leveraging innovative lightweight validation techniques and aligned incentives, we're making advanced computer vision accessible and scalable.

Our initial focus is Game State Recognition (GSR) in football - a strategic entry point into the $600 billion football industry, with $50 billion in betting and $30 billion in data services. Current solutions are prohibitively expensive: a single football match requires hundreds of hours of manual annotation, costing thousands of dollars. Score Vision aims to reduce these costs by 10x to 100x while dramatically improving speed and accuracy, unlocking new possibilities in sports analytics and beyond.

**Get Started:**

- [Miner Setup](miner/README.md)
- [Validator Setup](validator/README.md)

## Overview

Our framework revolutionises video analysis through a novel lightweight validation approach that ensures accuracy whilst minimising computational overhead. This innovation enables rapid, cost-effective processing of complex visual data at scale - a breakthrough for industries dependent on real-time video analysis.

### Why Football?

Football represents the perfect proving ground for our technology: high-stakes, real-time decision making, and massive global reach. With our team's deep connections across the sports data industry, we're uniquely positioned to deploy and scale our solution. However, this is just the beginning - our framework is designed to extend beyond sports into broader computer vision applications.

## Market Challenge

The sports video analysis market currently faces significant barriers to entry and scalability. Manual video annotation alone costs between $10-55 per minute, with complex sports scenarios demanding up to four hours of human labelling per minute of footage. For a single football match, this translates to major costs and low efficiency.

Traditional solutions struggle with fundamental technical limitations. Real-time processing capacity remains insufficient for livematch analysis. Accuracy suffers in dynamic environments with changing conditions and camera angles. Most systems lack seamless integration with existing sports analytics platforms.

The SoccerNet Game State Recognition (GSR) challenge establishes the current industry benchmark for football video analysis. Our framework aims to surpass these standards whilst dramatically reducing both operational costs and processing time.

Our solution serves four primary market segments. Professional, semi-pro and non-league clubs and leagues require comprehensive match analysis for performance optimisation. Broadcasters need real-time statistics to enhance viewer experience. Betting operators demand instant, accurate data for odds calculations. Analytics providers focus on delivering scalable insights across multiple matches.

## Architecture

Score Vision operates through a carefully orchestrated interaction between three key roles. Each role works in concert to enable efficient, decentralised video processing.

The video input stage begins with miners receiving video streams. Each frame undergoes processing for object detection and tracking, resulting in standardised outputs generated in real-time.

The validation stage follows, where validators verify miners’ outputs through filtered frame selection and hybrid scoring. Keypoints are assessed globally for stability and plausibility, while object detections undergo semantic verification, ensuring real-time, quality-aware feedback for network integrity.

Network management is overseen by Subnet Owner who maintain system health. They oversee the dynamic adjustment of incentives based on performance and optimise system parameters for maximum efficiency.

## Technical Implementation

### Core Innovation: Lightweight Validation

Our validation system combines precision and efficiency through a two-step process:
	1.	Frame Filtering & Keypoint Validation
Frames are filtered using pitch detection. A global scoring system then evaluates keypoint accuracy based on stability, plausibility, and reprojection error.
	2.	Semantic BBox Assessment
Selected frames are validated via CLIP-based object checks, verifying class accuracy for players, ball, referees, and goalkeepers. The result is a confidence-weighted quality score.

Scores are combined and normalized (0–1) to guide miner rewards and maintain network integrity


### Technical Challenges & Solutions

Real-time processing at scale presents a significant challenge when handling multiple high-definition video streams simultaneously. We solve this through distributed computation across miners with optimised frame sampling, enabling concurrent analysis of multiple matches with minimal latency.

Dynamic environments pose another substantial challenge. Rapid scene changes, varying camera angles, and weather conditions can impair analysis quality. Our solution combines robust object detection models with state-aware validation to maintain consistent accuracy across diverse viewing conditions.

Traditional validation requires extensive computational resources. Our approach implements smart frame sampling and progressive validation stages, achieving a tenfold reduction in validation costs whilst maintaining accuracy.

## Performance Metrics

### Core Performance Measurement

The Game State Higher Order Tracking Accuracy (GS-HOTA) metric forms the foundation of our quality assessment:

GS-HOTA = √(Detection × Association)

Detection measures object detection accuracy, whilst Association assesses tracking consistency across frames.

### Network Performance & Rewards

Miner evaluation encompasses quality scoring based on detection and tracking accuracy, consistency measurement through continuous performance across frames, and response time assessment. Reward distribution is weighted according to accuracy and contribution volume.

Validator assessment focuses on verification accuracy through consensus alignment, response efficiency in the validation process, and overall network contribution. Reward allocation reflects both validation accuracy and throughput.

Our dynamic incentive system adjusts rewards based on network demands and performance. High performers receive additional incentives, whilst poor performance or malicious behaviour incurs penalties.

## Roadmap

### Phase 1 (Current)

- [x] Game State Recognition challenge implementation
- [x] VLM-based validation
- [x] Incentive mechanism
- [x] Testnet deploy on netuid 261
- [x] Community testing
- [x] Comprehensive benchmarking

### Phase 2 (Q1 2025)

- [x] mainnet deploy on netuid 44 week commencing 13 January
- [ ] Human-in-the-loop validation
- [ ] Additional footage type (grassroots)
- [ ] Dashboard and Leaderboard

### Phase 3 (Q2-Q3 2025)

- [ ] Action spotting integration
- [ ] Match event captioning
- [ ] Advanced player tracking

### Phase 4 (Q4 2025)

- [ ] Integration APIs
- [ ] Additional sports adaptation
- [ ] Developer tools and SDKs
- [ ] Community contribution framework

### Future Developments

#### Action Spotting and Captioning

- Event detection (goals, fouls, etc.)
- Automated highlight generation
- Natural language descriptions

#### Cross-Domain Applications

- Basketball and tennis analysis
- Security surveillance
- Retail analytics

#### Technical Enhancements

- Advanced VLM capabilities
- Improved attribute assessment
- Adaptive learning mechanisms
- Open-source VLM development

## Research & Innovation

Our research paper, ["Score Vision: Enabling Complex Computer Vision Through Lightweight Validation - A Game State Recognition Framework for Live Football,"](https://drive.google.com/file/d/1oADURxxIZK0mTEqJPDudgXypohtFNkON/view) introduces groundbreaking approaches to computer vision validation.

Our lightweight validation technique dramatically reduces computational overhead and resource requirements, making advanced computer vision accessible at scale. Distributed processing combined with intelligent validation enables real-time analysis of multiple video streams. Novel validation mechanisms ensure high accuracy whilst reducing computational complexity. Our framework enables the collection of new data points at unprecedented scale, contributing to the advancement of computer vision research.

## Quick Start

To begin with Score Vision, ensure your system meets our technical requirements and complete the environment setup process. Our installation process is straightforward and can be completed in minutes using our command-line interface.

Detailed setup instructions for both miners and validators are available in our comprehensive setup guides. These guides provide step-by-step instructions for node configuration and network participation.

- [Miner Setup Guide](miner/README.md)
- [Validator Setup Guide](validator/README.md)

## Contributing

We welcome contributions to Score Vision. Our [Contributing Guidelines](CONTRIBUTING.md) provide comprehensive information about our code style standards, pull request process, development workflow, and testing requirements. We encourage both technical and non-technical contributions to our growing ecosystem.

## Community & Support

Join our vibrant community on Discord at [Score Vision Discord]. Follow our latest updates on Twitter[@webuildscore](https://x.com/webuildscore). For direct enquiries, reach us at hello@wearescore.com.

Our team combines extensive expertise in computer vision, sports technology, and distributed systems. We bring together leading minds in artificial intelligence, sports analytics, and blockchain technology to revolutionise video analysis.

## License

This project is licensed under the MIT License. Full details are available in the [LICENSE](LICENSE) file.
