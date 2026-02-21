/*
 * AIAudioBridgeControls.h - controls for AI audio bridge effect
 */

#ifndef LMMS_AI_AUDIO_BRIDGE_CONTROLS_H
#define LMMS_AI_AUDIO_BRIDGE_CONTROLS_H

#include "EffectControls.h"

#include "AIAudioBridgeControlDialog.h"
#include "AIAudioBridgeTransport.h"

namespace lmms
{

class AIAudioBridgeEffect;

class AIAudioBridgeControls : public EffectControls
{
	Q_OBJECT

public:
	explicit AIAudioBridgeControls(AIAudioBridgeEffect* effect);
	~AIAudioBridgeControls() override = default;

	void saveSettings(QDomDocument& doc, QDomElement& parent) override;
	void loadSettings(const QDomElement& parent) override;
	QString nodeName() const override
	{
		return "AIAudioBridgeControls";
	}
	gui::EffectControlDialog* createView() override
	{
		return new gui::AIAudioBridgeControlDialog(this);
	}
	int controlCount() override
	{
		return 5;
	}

	AIAudioBridgeTransport::Config transportConfig() const;

private:
	AIAudioBridgeEffect* m_effect = nullptr;

	BoolModel m_bridgeEnabledModel;
	FloatModel m_mixModel;
	FloatModel m_driveModel;
	IntModel m_transportModeModel;
	IntModel m_backendModel;

	QString m_command;
	QString m_scriptPath;
	QString m_apiKey;

	friend class gui::AIAudioBridgeControlDialog;
	friend class AIAudioBridgeEffect;
};

} // namespace lmms

#endif // LMMS_AI_AUDIO_BRIDGE_CONTROLS_H
