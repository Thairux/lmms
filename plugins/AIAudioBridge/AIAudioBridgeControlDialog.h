/*
 * AIAudioBridgeControlDialog.h - UI for AI audio bridge effect
 */

#ifndef LMMS_AI_AUDIO_BRIDGE_CONTROL_DIALOG_H
#define LMMS_AI_AUDIO_BRIDGE_CONTROL_DIALOG_H

#include "EffectControlDialog.h"

class QComboBox;
class QLineEdit;

namespace lmms
{
class AIAudioBridgeControls;
}

namespace lmms::gui
{

class AIAudioBridgeControlDialog : public EffectControlDialog
{
	Q_OBJECT

public:
	explicit AIAudioBridgeControlDialog(AIAudioBridgeControls* controls);
	~AIAudioBridgeControlDialog() override = default;

private:
	AIAudioBridgeControls* m_controls = nullptr;
	QComboBox* m_transportMode = nullptr;
	QComboBox* m_backend = nullptr;
	QLineEdit* m_command = nullptr;
	QLineEdit* m_scriptPath = nullptr;
	QLineEdit* m_apiKey = nullptr;
};

} // namespace lmms::gui

#endif // LMMS_AI_AUDIO_BRIDGE_CONTROL_DIALOG_H
