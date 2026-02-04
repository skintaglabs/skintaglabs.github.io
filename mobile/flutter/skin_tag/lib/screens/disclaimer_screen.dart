import 'package:flutter/material.dart';
import 'package:provider/provider.dart';
import '../main.dart';
import '../theme/app_theme.dart';

/// Medical disclaimer screen shown before first use
class DisclaimerScreen extends StatelessWidget {
  const DisclaimerScreen({super.key});

  @override
  Widget build(BuildContext context) {
    return Scaffold(
      body: SafeArea(
        child: Column(
          children: [
            Expanded(
              child: SingleChildScrollView(
                padding: const EdgeInsets.all(24),
                child: Column(
                  crossAxisAlignment: CrossAxisAlignment.center,
                  children: [
                    const SizedBox(height: 20),
                    // Header
                    const Icon(
                      Icons.medical_services,
                      size: 56,
                      color: AppColors.riskHigh,
                    ),
                    const SizedBox(height: 16),
                    Text(
                      'Important Medical Disclaimer',
                      style: Theme.of(context).textTheme.headlineSmall?.copyWith(
                            fontWeight: FontWeight.bold,
                          ),
                      textAlign: TextAlign.center,
                    ),
                    const SizedBox(height: 24),
                    const Divider(),
                    const SizedBox(height: 24),

                    // Disclaimer sections
                    _DisclaimerSection(
                      icon: Icons.warning_amber_rounded,
                      iconColor: Colors.orange,
                      title: 'This Is NOT a Medical Diagnosis',
                      content:
                          'SkinTag is an educational screening tool only. It does NOT provide medical diagnoses, medical advice, or treatment recommendations.',
                    ),
                    const SizedBox(height: 20),
                    _DisclaimerSection(
                      icon: Icons.local_hospital,
                      iconColor: AppColors.trustBlue,
                      title: 'Always Consult a Doctor',
                      content:
                          'Any skin lesion that concerns you should be evaluated by a qualified dermatologist or healthcare provider.',
                    ),
                    const SizedBox(height: 20),
                    _DisclaimerSection(
                      icon: Icons.psychology,
                      iconColor: Colors.purple,
                      title: 'AI Limitations',
                      content:
                          'This AI model has limitations and can make errors. False positives and false negatives are possible.',
                    ),
                    const SizedBox(height: 20),
                    _DisclaimerSection(
                      icon: Icons.emergency,
                      iconColor: Colors.teal,
                      title: 'Not for Emergency Use',
                      content:
                          'If you notice rapid changes, bleeding, or concerning symptoms, seek immediate medical attention.',
                    ),
                    const SizedBox(height: 24),

                    // Agreement text
                    Container(
                      padding: const EdgeInsets.all(16),
                      decoration: BoxDecoration(
                        color: AppColors.sageLight.withOpacity(0.3),
                        borderRadius: BorderRadius.circular(12),
                      ),
                      child: Text(
                        'By tapping "I Understand and Accept" below, you acknowledge that SkinTag is not a substitute for professional medical care.',
                        style: Theme.of(context).textTheme.bodySmall,
                        textAlign: TextAlign.center,
                      ),
                    ),
                    const SizedBox(height: 20),
                  ],
                ),
              ),
            ),

            // Accept button
            Container(
              padding: const EdgeInsets.all(16),
              decoration: BoxDecoration(
                color: Theme.of(context).scaffoldBackgroundColor,
                boxShadow: [
                  BoxShadow(
                    color: Colors.black.withOpacity(0.1),
                    blurRadius: 8,
                    offset: const Offset(0, -2),
                  ),
                ],
              ),
              child: SafeArea(
                top: false,
                child: SizedBox(
                  width: double.infinity,
                  child: ElevatedButton.icon(
                    onPressed: () {
                      context.read<AppState>().acceptDisclaimer();
                    },
                    icon: const Icon(Icons.check_circle),
                    label: const Text('I Understand and Accept'),
                    style: ElevatedButton.styleFrom(
                      padding: const EdgeInsets.symmetric(vertical: 16),
                    ),
                  ),
                ),
              ),
            ),
          ],
        ),
      ),
    );
  }
}

class _DisclaimerSection extends StatelessWidget {
  final IconData icon;
  final Color iconColor;
  final String title;
  final String content;

  const _DisclaimerSection({
    required this.icon,
    required this.iconColor,
    required this.title,
    required this.content,
  });

  @override
  Widget build(BuildContext context) {
    return Row(
      crossAxisAlignment: CrossAxisAlignment.start,
      children: [
        Icon(icon, size: 28, color: iconColor),
        const SizedBox(width: 16),
        Expanded(
          child: Column(
            crossAxisAlignment: CrossAxisAlignment.start,
            children: [
              Text(
                title,
                style: Theme.of(context).textTheme.titleMedium?.copyWith(
                      fontWeight: FontWeight.bold,
                    ),
              ),
              const SizedBox(height: 4),
              Text(
                content,
                style: Theme.of(context).textTheme.bodyMedium,
              ),
            ],
          ),
        ),
      ],
    );
  }
}
